# StereoMorph — Command Line Interface (CLI) Module

module CLI

using Printf
using ..Types
using ..Bounds
using ..Metrization
using ..Embedding
using ..Refinement
using ..Alignment
using ..PharmacophoreModel
using ..MolecularIO

export run_cli

function print_usage()
    println("""
StereoMorph.jl — High-Performance 3D Molecular Distance Geometry & Pharmacophore Engine

Usage:
    julia --project=. -m StereoMorph <command> [options]

Commands:
    generate <molecule>    Generate 3D conformations (presets: aspirin, ibuprofen, alanine, benzene, cyclohexane, butane)
                           Options: --samples <N> (default: 10), --cutoff <Å> (default: 0.6), --out <file.xyz/.sdf>
    align <file1> <file2>  Kabsch optimal rotational alignment between two 3D coordinate files and RMSD report
    pharmacophore <mol>    Extract 3D pharmacophore feature centroids and spatial distance matrix
    benchmark              Execute full performance and precision benchmark across standard chemical topologies
    help                   Show this usage manual
""")
end

function get_molecule_by_name(name::String)::Molecule
    lower = lowercase(name)
    if lower in ("aspirin", "acetylsalicylic_acid")
        return make_aspirin()
    elseif lower in ("ibuprofen", "advil")
        return make_ibuprofen()
    elseif lower in ("alanine", "l-alanine")
        return make_alanine(:L)
    elseif lower == "d-alanine"
        return make_alanine(:D)
    elseif lower in ("benzene", "c6h6")
        return make_benzene()
    elseif lower in ("cyclohexane", "c6h12")
        return make_cyclohexane()
    elseif lower in ("butane", "c4h10")
        return make_butane()
    elseif lower in ("propane", "c3h8")
        return make_propane()
    elseif lower in ("ethane", "c2h6")
        return make_ethane()
    elseif lower in ("methane", "ch4")
        return make_methane()
    else
        error("Unknown molecule name '$name'. Available presets: aspirin, ibuprofen, alanine, benzene, cyclohexane, butane, propane, ethane, methane.")
    end
end

function generate_conf_internal(mol::Molecule; method::Symbol=:lbfgs, max_iter::Int=300)::Conformation
    bm = compute_bounds(mol)
    D = crippen_havel_metrize(bm)
    coords, q = embed_distance_matrix(D)
    orient_chirality!(coords, mol)
    e, conv, _ = refine_conformation!(coords, mol, bm; max_iter=max_iter, method=method)
    v = count_violations(coords, mol, bm)
    return Conformation(coords, e, conv, q, v)
end

function run_cli(args::Vector{String}=ARGS)
    if isempty(args) || args[1] in ("-h", "--help", "help")
        print_usage()
        return 0
    end

    cmd = args[1]

    if cmd == "generate"
        if length(args) < 2
            println(stderr, "Error: missing molecule name.")
            print_usage()
            return 1
        end

        mol_name = args[2]
        mol = get_molecule_by_name(mol_name)

        # Parse options
        n_samples = 10
        cutoff = 0.6
        out_file = ""

        idx = 3
        while idx <= length(args)
            if args[idx] in ("-n", "--samples") && idx + 1 <= length(args)
                n_samples = parse(Int, args[idx + 1])
                idx += 2
            elseif args[idx] in ("-c", "--cutoff") && idx + 1 <= length(args)
                cutoff = parse(Float64, args[idx + 1])
                idx += 2
            elseif args[idx] in ("-o", "--out") && idx + 1 <= length(args)
                out_file = args[idx + 1]
                idx += 2
            else
                idx += 1
            end
        end

        println("======================================================================")
        println(" StereoMorph.jl — Distance Geometry 3D Conformation Generator")
        println(" Target Molecule: $(mol.name) ($(length(mol.atoms)) atoms, $(length(mol.bonds)) bonds)")
        println(" Samples Requested: $n_samples | Clustering Cutoff: $cutoff Å")
        println("======================================================================")

        t_start = time()
        bm = compute_bounds(mol)
        t_bounds = time() - t_start
        @printf("✓ Bounds computed and triangle smoothed in %.4f s\n", t_bounds)

        all_confs = Conformation[]
        for s in 1:n_samples
            c = generate_conf_internal(mol; method=:lbfgs, max_iter=300)
            push!(all_confs, c)
        end
        t_gen = time() - t_start - t_bounds
        @printf("✓ %d conformations generated and refined in %.4f s (%.2f ms/conf)\n",
                n_samples, t_gen, (t_gen / n_samples) * 1000)

        # Clustering
        reps, assignments = cluster_conformers(all_confs; cutoff=cutoff)
        @printf("✓ Extracted %d distinct conformational clusters (cutoff = %.2f Å)\n", length(reps), cutoff)
        println("----------------------------------------------------------------------")
        @printf("%-10s | %-16s | %-12s | %-12s\n", "Cluster #", "Energy (kcal/mol)", "Quality Q_3D", "Violations")
        println("----------------------------------------------------------------------")
        for (i, r) in enumerate(reps)
            @printf("Cluster %-2d | %16.4f | %12.4f | %-12d\n", i, r.energy, r.embedding_quality, r.violations)
        end
        println("----------------------------------------------------------------------")

        if !isempty(out_file)
            if endswith(out_file, ".sdf") || endswith(out_file, ".mol")
                write_sdf(out_file, mol, reps[1].coords, mol.name)
                println("✓ Lowest-energy conformer saved to $out_file (MDL V2000 format)")
            else
                write_ensemble_xyz(out_file, mol, reps)
                println("✓ Unique conformer ensemble saved to $out_file (Multi-frame XYZ format)")
            end
        end

    elseif cmd == "align"
        if length(args) < 3
            println(stderr, "Error: align requires two XYZ coordinate files: align <file1.xyz> <file2.xyz>")
            return 1
        end

        file1, file2 = args[2], args[3]
        syms1, coords1 = read_xyz(file1)
        syms2, coords2 = read_xyz(file2)

        if length(syms1) != length(syms2)
            println(stderr, "Error: coordinate files have different atom counts ($(length(syms1)) vs $(length(syms2))).")
            return 1
        end

        aligned_p, rot_mat, rmsd_val = align_coordinates(coords1, coords2)
        println("======================================================================")
        println(" StereoMorph.jl — Kabsch Rotational Alignment & Superposition")
        println(" File 1: $file1 | File 2: $file2 | Atom Count: $(length(syms1))")
        println("======================================================================")
        @printf("Optimal Kabsch RMSD: %.5f Å\n", rmsd_val)
        println("3x3 Proper Rotation Matrix R (det = +1.0):")
        for r in 1:3
            @printf("  [ %9.6f  %9.6f  %9.6f ]\n", rot_mat[r, 1], rot_mat[r, 2], rot_mat[r, 3])
        end
        println("----------------------------------------------------------------------")

    elseif cmd == "pharmacophore"
        if length(args) < 2
            println(stderr, "Error: missing molecule name.")
            return 1
        end

        mol_name = args[2]
        mol = get_molecule_by_name(mol_name)
        conf = generate_conf_internal(mol; method=:lbfgs)
        ph = extract_pharmacophore(mol, conf.coords)

        println("======================================================================")
        println(" StereoMorph.jl — 3D Pharmacophore Feature Extraction")
        println(" Target: $(mol.name)")
        println("======================================================================")
        print_pharmacophore_summary(stdout, ph)

    elseif cmd == "benchmark"
        println("======================================================================")
        println(" StereoMorph.jl — Comprehensive Benchmark Suite")
        println("======================================================================")

        benchmarks = [
            ("Methane", make_methane()),
            ("Ethane", make_ethane()),
            ("Propane", make_propane()),
            ("Butane", make_butane()),
            ("Cyclohexane", make_cyclohexane()),
            ("Benzene", make_benzene()),
            ("L-Alanine", make_alanine(:L)),
            ("Aspirin", make_aspirin()),
            ("Ibuprofen", make_ibuprofen())
        ]

        @printf("%-14s | %-6s | %-6s | %-12s | %-12s | %-10s\n",
                "Molecule", "Atoms", "Bonds", "Time (ms)", "Energy", "Violations")
        println("----------------------------------------------------------------------")

        for (name, mol) in benchmarks
            t0 = time()
            conf = generate_conf_internal(mol; max_iter=250)
            elapsed_ms = (time() - t0) * 1000.0

            @printf("%-14s | %-6d | %-6d | %12.2f | %12.4f | %-10d\n",
                    name, length(mol.atoms), length(mol.bonds), elapsed_ms, conf.energy, conf.violations)
        end
        println("======================================================================")
    else
        println(stderr, "Error: unknown command '$cmd'.")
        print_usage()
        return 1
    end

    return 0
end

end # module CLI
