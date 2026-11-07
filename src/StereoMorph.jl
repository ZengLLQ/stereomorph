# StereoMorph.jl — High-Performance 3D Molecular Distance Geometry,
# Conformation Embedding & Pharmacophore Alignment Engine

module StereoMorph

using LinearAlgebra
using Statistics
using Random
using Printf

# Submodules
include("types.jl")
include("bounds.jl")
include("metrization.jl")
include("embedding.jl")
include("refinement.jl")
include("alignment.jl")
include("pharmacophore.jl")
include("io.jl")
include("cli.jl")

# Re-export submodules
using .Types
using .Bounds
using .Metrization
using .Embedding
using .Refinement
using .Alignment
using .PharmacophoreModel
using .MolecularIO
using .CLI

export Types, Bounds, Metrization, Embedding, Refinement, Alignment, PharmacophoreModel, MolecularIO, CLI

# Re-export core types
export Atom, Bond, ChiralCenter, PlanarSystem, Molecule, Conformation
export Hybridization, SP3, SP2, SP, UNKNOWN_HYB
export ChiralSign, R_CHIRAL, S_CHIRAL, ACHIRAL
export BoundsMatrix, FeaturePoint, Pharmacophore

# Re-export key functions
export generate_conformation, generate_ensemble
export compute_bounds, smooth_bounds!, sample_trial_distances, crippen_havel_metrize
export check_bounds_consistency, BoundsInconsistencyError, shortest_path_matrix, find_rings
export cayley_menger_det4, cayley_menger_det5
export embed_distance_matrix, orient_chirality!, signed_chiral_volume
export refine_conformation!, count_violations, evaluate_energy_and_gradient
export calculate_rmsd, align_coordinates, pairwise_rmsd_matrix, cluster_conformers
export extract_pharmacophore, score_pharmacophore_overlay, print_pharmacophore_summary
export write_xyz, read_xyz, write_sdf, read_mol, write_ensemble_xyz
export make_methane, make_ethane, make_propane, make_butane, make_cyclohexane
export make_benzene, make_aspirin, make_alanine, make_ibuprofen

"""
    generate_conformation(mol::Molecule;
                          method::Symbol=:lbfgs,
                          max_iter::Int=300,
                          use_metrization::Bool=true,
                          rng::AbstractRNG=Random.GLOBAL_RNG)::Conformation

Generates a single refined 3D molecular conformation from topological description:
1. Computes initial covalent, geminal, and non-bonded distance bounds.
2. Applies Floyd-Warshall triangle inequality metric smoothing.
3. Performs Crippen-Havel incremental metrization or random bounds sampling.
4. Projects metric Gram matrix into 3D via Classical Multidimensional Scaling (MDS).
5. Enforces stereochemical parity at chiral centers.
6. Minimizes distance geometry penalty energy using L-BFGS or Conjugate Gradient.
"""
function generate_conformation(mol::Molecule;
                              method::Symbol=:lbfgs,
                              max_iter::Int=300,
                              use_metrization::Bool=true,
                              rng::AbstractRNG=Random.GLOBAL_RNG)::Conformation
    # 1. Compute and smooth distance bounds
    bm = compute_bounds(mol)

    # 2. Sample trial distance matrix
    D = if use_metrization
        crippen_havel_metrize(bm; rng=rng)
    else
        sample_trial_distances(bm; rng=rng)
    end

    # 3. Metric matrix embedding (Classical MDS)
    coords, quality = embed_distance_matrix(D)

    # 4. Check and orient chiral centers
    orient_chirality!(coords, mol)

    # 5. Energy minimization / force field refinement
    final_energy, converged, _ = refine_conformation!(coords, mol, bm;
                                                     max_iter=max_iter,
                                                     method=method)

    # 6. Count remaining violations
    violations = count_violations(coords, mol, bm)

    return Conformation(coords, final_energy, converged, quality, violations)
end

"""
    generate_ensemble(mol::Molecule, n_confs::Int=10;
                      method::Symbol=:lbfgs,
                      max_iter::Int=300,
                      cluster_cutoff::Float64=0.6,
                      use_metrization::Bool=true,
                      rng::AbstractRNG=Random.GLOBAL_RNG)::Tuple{Vector{Conformation}, Vector{Conformation}}

Generates an ensemble of n_confs 3D conformations and performs RMSD-based clustering
to yield distinct conformational isomers.
Returns (all_conformers, unique_cluster_representatives).
"""
function generate_ensemble(mol::Molecule, n_confs::Int=10;
                          method::Symbol=:lbfgs,
                          max_iter::Int=300,
                          cluster_cutoff::Float64=0.6,
                          use_metrization::Bool=true,
                          rng::AbstractRNG=Random.GLOBAL_RNG)::Tuple{Vector{Conformation}, Vector{Conformation}}
    all_confs = Conformation[]
    for i in 1:n_confs
        c = generate_conformation(mol; method=method, max_iter=max_iter,
                                  use_metrization=use_metrization, rng=rng)
        push!(all_confs, c)
    end

    representatives, _ = cluster_conformers(all_confs; cutoff=cluster_cutoff)
    return all_confs, representatives
end

end # module StereoMorph
