# StereoMorph.jl — Kabsch Alignment & RMSD Demo

using StereoMorph
using Printf

println("===================================================================")
println(" StereoMorph.jl — Kabsch Algorithm & Superposition Demo")
println("===================================================================")

ibuprofen = make_ibuprofen()
println("Molecule: $(ibuprofen.name) ($(length(ibuprofen.atoms)) atoms)")

# Generate two distinct conformations
conf1 = generate_conformation(ibuprofen; max_iter=250)
conf2 = generate_conformation(ibuprofen; max_iter=250)

# Unaligned raw RMSD
raw_diff = conf1.coords .- conf2.coords
raw_rmsd = sqrt(sum(raw_diff .^ 2) / length(ibuprofen.atoms))
@printf("Unaligned Raw Coordinate RMSD: %.5f Å\n", raw_rmsd)

# Optimal Kabsch Alignment
aligned_coords, rot_matrix, opt_rmsd = align_coordinates(conf1.coords, conf2.coords)
@printf("Optimal Kabsch-Superimposed RMSD: %.5f Å\n", opt_rmsd)

println("\nCalculated 3x3 Proper Rotation Matrix:")
for r in 1:3
    @printf("  [ %9.6f  %9.6f  %9.6f ]\n", rot_matrix[r, 1], rot_matrix[r, 2], rot_matrix[r, 3])
end

@printf("\nSuperposition reduced structural divergence from %.4f Å to %.4f Å (%.1f%% reduction).\n",
        raw_rmsd, opt_rmsd, (1.0 - opt_rmsd / raw_rmsd) * 100.0)
println("===================================================================")
