# StereoMorph — 3D Pharmacophore Feature Extraction & Alignment Module

module PharmacophoreModel

using LinearAlgebra
using Statistics: mean
using Printf
using ..Types

export FeaturePoint, Pharmacophore
export extract_pharmacophore, score_pharmacophore_overlay, print_pharmacophore_summary

"""
    FeaturePoint

A 3D pharmacophore feature point with spatial coordinates, directional vector, and atom indices.
Types: :HBD (Donor), :HBA (Acceptor), :AROM (Aromatic), :HYD (Hydrophobic), :POS (Positive), :NEG (Negative).
"""
struct FeaturePoint
    type::Symbol
    center::Vector{Float64}
    direction::Vector{Float64}
    atom_indices::Vector{Int}
    name::String
end

"""
    Pharmacophore

Collection of 3D feature points and their inter-feature Euclidean distance matrix.
"""
struct Pharmacophore
    features::Vector{FeaturePoint}
    distance_matrix::Matrix{Float64}

    function Pharmacophore(features::Vector{FeaturePoint})
        k = length(features)
        mat = zeros(Float64, k, k)
        for i in 1:k
            for j in (i + 1):k
                d = norm(features[i].center .- features[j].center)
                mat[i, j] = mat[j, i] = d
            end
        end
        new(features, mat)
    end
end

"""
    extract_pharmacophore(mol::Molecule, coords::Matrix{Float64})::Pharmacophore

Identifies 3D pharmacophore features (HBD, HBA, AROM, HYD, POS, NEG) from molecular topology
and assigned 3D Cartesian coordinates.
"""
function extract_pharmacophore(mol::Molecule, coords::Matrix{Float64})::Pharmacophore
    features = FeaturePoint[]
    n = length(mol.atoms)

    # 1. Hydrogen Bond Donors (HBD) & Acceptors (HBA)
    for i in 1:n
        sym = mol.atoms[i].symbol
        c = coords[:, i]

        if sym == "O"
            # Oxygen: donor if bonded to H, or if it is an OH group (single-bonded to C with no other heavy bonds)
            has_h = any(mol.atoms[nbr].symbol == "H" for nbr in mol.adj_list[i])
            is_hydroxyl = (length(mol.adj_list[i]) == 1 && mol.atoms[mol.adj_list[i][1]].symbol == "C" && mol.atoms[i].hybridization == SP3)
            if has_h || is_hydroxyl
                # HBD feature
                dir = zeros(Float64, 3)
                if has_h
                    for nbr in mol.adj_list[i]
                        if mol.atoms[nbr].symbol == "H"
                            dir = coords[:, nbr] .- c
                            if norm(dir) > 1e-4 dir ./= norm(dir) end
                        end
                    end
                else
                    c_nbr = coords[:, mol.adj_list[i][1]]
                    dir = c .- c_nbr
                    if norm(dir) > 1e-4 dir ./= norm(dir) end
                end
                push!(features, FeaturePoint(:HBD, c, dir, [i], "Hydroxyl Donor (O$i)"))
            end

            # All oxygens also act as HBA (lone pairs)
            push!(features, FeaturePoint(:HBA, c, [0.0, 0.0, 0.0], [i], "Oxygen Acceptor (O$i)"))
        elseif sym == "N"
            has_h = any(mol.atoms[nbr].symbol == "H" for nbr in mol.adj_list[i])
            if has_h
                push!(features, FeaturePoint(:HBD, c, [0.0, 0.0, 1.0], [i], "Amine/Amide Donor (N$i)"))
            end
            push!(features, FeaturePoint(:HBA, c, [0.0, 0.0, 0.0], [i], "Nitrogen Acceptor (N$i)"))
        end
    end

    # 2. Aromatic Rings (AROM)
    rings = find_rings(mol)
    for r in rings
        if length(r) == 6
            # Check if all atoms in ring are aromatic or SP2
            all_arom = all(mol.atoms[idx].hybridization == SP2 for idx in r)
            if all_arom
                centroid = [mean(coords[1, r]), mean(coords[2, r]), mean(coords[3, r])]
                # Compute normal vector
                p1 = coords[:, r[1]] .- centroid
                p2 = coords[:, r[2]] .- centroid
                n_vec = [p1[2]*p2[3] - p1[3]*p2[2], p1[3]*p2[1] - p1[1]*p2[3], p1[1]*p2[2] - p1[2]*p2[1]]
                if norm(n_vec) > 1e-4
                    n_vec ./= norm(n_vec)
                end
                push!(features, FeaturePoint(:AROM, centroid, n_vec, r, "Aromatic Ring (6-ring)"))
            end
        end
    end

    # 3. Hydrophobic Centers (HYD) - Aliphatic alkyl groups
    for i in 1:n
        atom = mol.atoms[i]
        if atom.symbol == "C" && atom.hybridization == SP3
            # Check if bonded only to C and H
            only_ch = all(mol.atoms[nbr].symbol in ("C", "H") for nbr in mol.adj_list[i])
            if only_ch
                c = coords[:, i]
                push!(features, FeaturePoint(:HYD, c, [0.0, 0.0, 0.0], [i], "Hydrophobic Carbon (C$i)"))
            end
        end
    end

    # 4. Acidic / Basic ionizable groups (NEG / POS)
    for ps in mol.planar_systems
        if occursin("Carboxyl", ps.description)
            # Find carboxyl carbon and oxygens
            c_idx = filter(idx -> mol.atoms[idx].symbol == "C", ps.atoms)
            o_indices = filter(idx -> mol.atoms[idx].symbol == "O", ps.atoms)
            if !isempty(o_indices)
                centroid = [mean(coords[1, o_indices]), mean(coords[2, o_indices]), mean(coords[3, o_indices])]
                push!(features, FeaturePoint(:NEG, centroid, [0.0, 0.0, 0.0], o_indices, "Carboxylate Anion"))
            end
        end
    end

    return Pharmacophore(features)
end

"""
    score_pharmacophore_overlay(query::Pharmacophore, target::Pharmacophore;
                                tolerance::Float64=1.5,
                                sigma::Float64=0.8)::Tuple{Float64, Vector{Tuple{Int,Int}}}

Calculates the 3D spatial alignment fit score between two pharmacophores.
Uses a greedy bipartite matching to pair features of identical types with distances < tolerance.
Returns (fit_score, matched_pairs).
Score is calculated via Gaussian overlap: S = sum(exp(-d_ij^2 / (2 * sigma^2))).
"""
function score_pharmacophore_overlay(query::Pharmacophore, target::Pharmacophore;
                                     tolerance::Float64=1.5,
                                     sigma::Float64=0.8)::Tuple{Float64, Vector{Tuple{Int,Int}}}
    q_feats = query.features
    t_feats = target.features

    matched_pairs = Tuple{Int,Int}[]
    used_targets = Set{Int}()
    fit_score = 0.0

    for (q_idx, qf) in enumerate(q_feats)
        best_t_idx = 0
        best_dist = Inf

        for (t_idx, tf) in enumerate(t_feats)
            if t_idx in used_targets continue end
            if qf.type != tf.type continue end

            dist = norm(qf.center .- tf.center)
            if dist < tolerance && dist < best_dist
                best_dist = dist
                best_t_idx = t_idx
            end
        end

        if best_t_idx != 0
            push!(used_targets, best_t_idx)
            push!(matched_pairs, (q_idx, best_t_idx))
            # Gaussian similarity contribution
            fit_score += exp(-best_dist^2 / (2.0 * sigma^2))
        end
    end

    return fit_score, matched_pairs
end

"""
    print_pharmacophore_summary(io::IO, ph::Pharmacophore)

Prints a tabular overview of pharmacophore features and their 3D coordinates.
"""
function print_pharmacophore_summary(io::IO, ph::Pharmacophore)
    println(io, "3D Pharmacophore Feature Summary ($(length(ph.features)) features):")
    println(io, "----------------------------------------------------------------------")
    @printf(io, "%-4s | %-6s | %-24s | %-22s\n", "#", "Type", "Name", "Coordinates (x, y, z) Å")
    println(io, "----------------------------------------------------------------------")
    for (idx, f) in enumerate(ph.features)
        coords_str = @sprintf("(%6.2f, %6.2f, %6.2f)", f.center[1], f.center[2], f.center[3])
        @printf(io, "%-4d | %-6s | %-24s | %-22s\n", idx, string(f.type), f.name, coords_str)
    end
    println(io, "----------------------------------------------------------------------")
end

end # module PharmacophoreModel
