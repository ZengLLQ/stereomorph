# StereoMorph.jl — 3D Conformation Generation Demo

using StereoMorph
using Printf

println("===================================================================")
println(" StereoMorph.jl — 3D Conformation Generation & Ensemble Clustering")
println("===================================================================")

aspirin = make_aspirin()
println("Target: $(aspirin.name) (Atoms: $(length(aspirin.atoms)), Bonds: $(length(aspirin.bonds)))")

println("\n1. Generating single refined 3D conformation with L-BFGS...")
conf = generate_conformation(aspirin; method=:lbfgs, max_iter=300)
@printf("   Final Penalty Energy:  %.6f kcal/mol\n", conf.energy)
@printf("   3D Embedding Quality:  %.4f\n", conf.embedding_quality)
@printf("   Remaining Violations:  %d\n", conf.violations)
@printf("   Optimizer Converged:   %s\n", conf.converged ? "YES" : "NO")

println("\n2. Generating ensemble of 15 conformations & clustering...")
all_confs, unique_reps = generate_ensemble(aspirin, 15; cluster_cutoff=0.6)
@printf("   Total Generated:       %d\n", length(all_confs))
@printf("   Unique Clusters Found: %d (cutoff = 0.60 Å)\n", length(unique_reps))

println("\n3. Cluster Representatives:")
for (i, rep) in enumerate(unique_reps)
    @printf("   Cluster %2d: Energy = %10.4f kcal/mol | Violations = %d\n",
            i, rep.energy, rep.violations)
end

# Save lowest-energy conformer to XYZ and SDF
output_xyz = "aspirin_3d.xyz"
output_sdf = "aspirin_3d.sdf"
write_xyz(output_xyz, aspirin, unique_reps[1].coords, "Aspirin Lowest Energy Conformer")
write_sdf(output_sdf, aspirin, unique_reps[1].coords, "Aspirin")
println("\n✓ Saved conformer to $output_xyz and $output_sdf")
println("===================================================================")
