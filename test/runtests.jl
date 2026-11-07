# StereoMorph.jl — Comprehensive Test Suite

using Test
using LinearAlgebra
using Statistics
using Random

using StereoMorph

@testset "StereoMorph.jl Test Suite" begin

    @testset "1. Types & Molecular Topology" begin
        methane = make_methane()
        @test length(methane.atoms) == 5
        @test length(methane.bonds) == 4
        @test methane.name == "Methane"

        ethane = make_ethane()
        @test length(ethane.atoms) == 8
        @test length(ethane.bonds) == 7

        sp_ethane = shortest_path_matrix(ethane)
        @test size(sp_ethane) == (8, 8)
        @test sp_ethane[1, 1] == 0
        @test sp_ethane[1, 2] == 1  # C-C bond
        @test sp_ethane[3, 6] == 3  # H-C-C-H

        # Ring detection
        benzene = make_benzene()
        rings_b = find_rings(benzene)
        @test length(rings_b) == 1
        @test length(rings_b[1]) == 6

        cyclohexane = make_cyclohexane()
        rings_c = find_rings(cyclohexane)
        @test length(rings_c) == 1
        @test length(rings_c[1]) == 6

        aspirin = make_aspirin()
        @test length(aspirin.atoms) == 13
        @test length(aspirin.bonds) == 13
        rings_a = find_rings(aspirin)
        @test length(rings_a) == 1

        ibuprofen = make_ibuprofen()
        @test length(ibuprofen.atoms) == 16
        @test length(ibuprofen.chiral_centers) == 1
        @test ibuprofen.chiral_centers[1].center == 11
    end

    @testset "2. Distance Bounds & Triangle Inequality Smoothing" begin
        propane = make_propane()
        bm = compute_bounds(propane)
        @test bm.n_atoms == 3
        @test check_bounds_consistency(bm)

        # C-C bond bounds
        @test 1.35 < bm.lower[1, 2] < 1.55
        @test 1.45 < bm.upper[1, 2] < 1.65
        @test bm.lower[1, 2] <= bm.upper[1, 2]

        # 1-3 C-C-C bounds (tetrahedral angle ~109.5 deg)
        @test bm.lower[1, 3] < bm.upper[1, 3]
        @test 2.3 < bm.upper[1, 3] < 2.7

        # Test triangle inequality satisfaction on Aspirin
        aspirin = make_aspirin()
        bm_asp = compute_bounds(aspirin)
        @test check_bounds_consistency(bm_asp)

        n = bm_asp.n_atoms
        for k in 1:n
            for i in 1:n
                for j in 1:n
                    @test bm_asp.upper[i, j] <= bm_asp.upper[i, k] + bm_asp.upper[k, j] + 1e-6
                    @test bm_asp.lower[i, j] >= bm_asp.lower[i, k] - bm_asp.upper[k, j] - 1e-6
                end
            end
        end

        # Inconsistency detection
        bad_bm = BoundsMatrix(3)
        bad_bm.lower[1, 2] = bad_bm.lower[2, 1] = 5.0
        bad_bm.upper[1, 2] = bad_bm.upper[2, 1] = 2.0
        @test_throws BoundsInconsistencyError check_bounds_consistency(bad_bm)
    end

    @testset "3. Metrization & Trial Sampling" begin
        benzene = make_benzene()
        bm = compute_bounds(benzene)

        # Uniform trial distance sampling
        D_trial = sample_trial_distances(bm; rng=MersenneTwister(42))
        @test size(D_trial) == (6, 6)
        @test D_trial ≈ D_trial'
        @test all(diag(D_trial) .== 0.0)

        for i in 1:6
            for j in (i + 1):6
                @test bm.lower[i, j] <= D_trial[i, j] <= bm.upper[i, j]
            end
        end

        # Crippen-Havel incremental metrization
        D_ch = crippen_havel_metrize(bm; rng=MersenneTwister(123))
        @test size(D_ch) == (6, 6)
        @test D_ch ≈ D_ch'
        for i in 1:6
            for j in (i + 1):6
                @test bm.lower[i, j] - 1e-4 <= D_ch[i, j] <= bm.upper[i, j] + 1e-4
            end
        end

        # Cayley-Menger determinants
        # Construct exact regular tetrahedron with edge a = 2.0
        a = 2.0
        D_tet = fill(a, 4, 4)
        for i in 1:4 D_tet[i, i] = 0.0 end
        cm4 = cayley_menger_det4(D_tet, 1, 2, 3, 4)
        # Volume V = a^3 / (6 * sqrt(2)) => V^2 = a^6 / 72
        # CM4 = 288 * V^2 = 288 * (64 / 72) = 256.0
        @test isapprox(cm4, 256.0, atol=1e-5)
    end

    @testset "4. Classical MDS Embedding" begin
        # Test embedding of known 3D points
        true_pts = [
            0.0  1.0  0.0  0.0;
            0.0  0.0  1.0  0.0;
            0.0  0.0  0.0  1.0
        ]
        n = size(true_pts, 2)
        D_true = zeros(Float64, n, n)
        for i in 1:n
            for j in 1:n
                D_true[i, j] = norm(true_pts[:, i] .- true_pts[:, j])
            end
        end

        coords, q = embed_distance_matrix(D_true)
        @test size(coords) == (3, n)
        @test q > 0.999 # Perfect 3D embeddability

        # Verify pairwise reconstructed distances match input distances
        for i in 1:n
            for j in (i + 1):n
                d_rec = norm(coords[:, i] .- coords[:, j])
                @test isapprox(d_rec, D_true[i, j], atol=1e-4)
            end
        end

        # Signed chiral volume
        v_pos = signed_chiral_volume(true_pts, 4, 1, 2, 3)
        @test abs(v_pos) > 0.1
    end

    @testset "5. Force Field & Penalty Energy Refinement" begin
        aspirin = make_aspirin()
        bm = compute_bounds(aspirin)
        D = sample_trial_distances(bm; rng=MersenneTwister(99))
        coords, _ = embed_distance_matrix(D)

        initial_energy, init_grad = evaluate_energy_and_gradient(coords, aspirin, bm)
        @test size(init_grad) == (3, length(aspirin.atoms))

        # Numerical gradient verification (finite difference check on atom 1)
        eps = 1e-6
        coords_plus = copy(coords)
        coords_plus[1, 1] += eps
        e_plus, _ = evaluate_energy_and_gradient(coords_plus, aspirin, bm)

        coords_minus = copy(coords)
        coords_minus[1, 1] -= eps
        e_minus, _ = evaluate_energy_and_gradient(coords_minus, aspirin, bm)

        num_grad_x1 = (e_plus - e_minus) / (2.0 * eps)
        @test isapprox(init_grad[1, 1], num_grad_x1, atol=1e-3, rtol=1e-3)

        # L-BFGS refinement
        final_energy, converged, iters = refine_conformation!(coords, aspirin, bm;
                                                             max_iter=300, method=:lbfgs)
        @test final_energy < initial_energy
        @test final_energy < 5.0 # Greatly minimized penalty
        @test iters > 0

        # Conjugate Gradient refinement
        coords_cg, _ = embed_distance_matrix(D)
        e_cg, _, _ = refine_conformation!(coords_cg, aspirin, bm;
                                         max_iter=250, method=:cg)
        @test e_cg < initial_energy
    end

    @testset "6. Alignment, Kabsch Algorithm & Clustering" begin
        # 1. Kabsch exact rotation recovery
        P = randn(MersenneTwister(7), 3, 10)
        # Random proper rotation matrix via Rodrigues formula
        axis = [0.0, 1.0, 0.0]
        theta = pi / 3.0 # 60 degrees
        K = [0.0 -axis[3] axis[2]; axis[3] 0.0 -axis[1]; -axis[2] axis[1] 0.0]
        R_true = Matrix{Float64}(I, 3, 3) + sin(theta) * K + (1.0 - cos(theta)) * (K * K)

        shift = [2.5, -1.0, 4.0]
        Q = (R_true * P) .+ shift

        aligned_P, R_calc, rmsd_val = align_coordinates(P, Q)
        @test rmsd_val < 1e-8
        @test isapprox(det(R_calc), 1.0, atol=1e-6)
        @test isapprox(R_calc * R_calc', Matrix{Float64}(I, 3, 3), atol=1e-6)

        # 2. Conformer clustering
        butane = make_butane()
        bm = compute_bounds(butane)

        confs = Conformation[]
        for seed in 1:6
            D = sample_trial_distances(bm; rng=MersenneTwister(seed * 10))
            c, q = embed_distance_matrix(D)
            e, conv, _ = refine_conformation!(c, butane, bm; max_iter=200)
            push!(confs, Conformation(c, e, conv, q, 0))
        end

        p_mat = pairwise_rmsd_matrix(confs)
        @test size(p_mat) == (6, 6)
        @test p_mat ≈ p_mat'
        @test all(diag(p_mat) .== 0.0)

        reps, assignments = cluster_conformers(confs; cutoff=0.5)
        @test length(reps) >= 1
        @test length(assignments) == 6
        @test all(assignments .>= 1)
    end

    @testset "7. 3D Pharmacophore Feature Extraction" begin
        aspirin = make_aspirin()
        conf = generate_conformation(aspirin; max_iter=250)

        ph = extract_pharmacophore(aspirin, conf.coords)
        @test length(ph.features) >= 4

        types_found = [f.type for f in ph.features]
        @test :AROM in types_found # Benzene ring
        @test :HBA in types_found  # Carbonyl/ester oxygens
        @test :HBD in types_found  # Carboxylic acid -OH

        @test size(ph.distance_matrix) == (length(ph.features), length(ph.features))
        @test ph.distance_matrix ≈ ph.distance_matrix'

        # Overlay scoring between identical pharmacophores
        score_self, pairs_self = score_pharmacophore_overlay(ph, ph; tolerance=0.1)
        @test length(pairs_self) == length(ph.features)
        @test score_self ≈ Float64(length(ph.features)) atol=1e-4

        # Test summary printer
        buf = IOBuffer()
        print_pharmacophore_summary(buf, ph)
        summary_str = String(take!(buf))
        @test occursin("3D Pharmacophore Feature Summary", summary_str)
        @test occursin("AROM", summary_str)
    end

    @testset "8. Molecular File I/O (XYZ & SDF)" begin
        alanine = make_alanine(:L)
        conf = generate_conformation(alanine; max_iter=250)

        # Test XYZ round-trip
        xyz_file = tempname() * ".xyz"
        write_xyz(xyz_file, alanine, conf.coords, "L-Alanine test")
        @test isfile(xyz_file)

        syms, coords_read = read_xyz(xyz_file)
        @test length(syms) == length(alanine.atoms)
        @test coords_read ≈ conf.coords atol=1e-5
        rm(xyz_file, force=true)

        # Test SDF round-trip
        sdf_file = tempname() * ".sdf"
        write_sdf(sdf_file, alanine, conf.coords, "L-Alanine")
        @test isfile(sdf_file)

        mol_read, coords_mol = read_mol(sdf_file)
        @test length(mol_read.atoms) == length(alanine.atoms)
        @test length(mol_read.bonds) == length(alanine.bonds)
        @test coords_mol ≈ conf.coords atol=1e-3
        rm(sdf_file, force=true)
    end

    @testset "9. End-to-End Benchmark & Robustness" begin
        # Test full pipeline on all standard chemical classes
        molecules = [
            make_methane(),
            make_ethane(),
            make_cyclohexane(),
            make_benzene(),
            make_alanine(:L),
            make_aspirin(),
            make_ibuprofen()
        ]

        for mol in molecules
            conf = generate_conformation(mol; method=:lbfgs, max_iter=300)
            @test conf.energy < 5.0
            @test conf.embedding_quality > 0.60
            @test conf.violations <= 15
            @test size(conf.coords) == (3, length(mol.atoms))
        end

        # Ensemble generation on Ibuprofen
        ibu = make_ibuprofen()
        all_confs, reps = generate_ensemble(ibu, 5; cluster_cutoff=0.5)
        @test length(all_confs) == 5
        @test length(reps) >= 1
    end

end # testset
