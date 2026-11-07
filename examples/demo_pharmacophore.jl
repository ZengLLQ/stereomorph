# StereoMorph.jl — 3D Pharmacophore Feature Extraction & Alignment Demo

using StereoMorph
using LinearAlgebra: norm
using Printf

println("===================================================================")
println(" StereoMorph.jl — 3D Pharmacophore Extraction & Overlay Demo")
println("===================================================================")

aspirin = make_aspirin()
ibuprofen = make_ibuprofen()

conf_asp = generate_conformation(aspirin; max_iter=250)
conf_ibu = generate_conformation(ibuprofen; max_iter=250)

ph_asp = extract_pharmacophore(aspirin, conf_asp.coords)
ph_ibu = extract_pharmacophore(ibuprofen, conf_ibu.coords)

println("\n--- Aspirin Pharmacophore ---")
print_pharmacophore_summary(stdout, ph_asp)

println("\n--- Ibuprofen Pharmacophore ---")
print_pharmacophore_summary(stdout, ph_ibu)

# Perform 3D pharmacophore overlay scoring
fit_score, matched_pairs = score_pharmacophore_overlay(ph_asp, ph_ibu; tolerance=2.0)

println("\n--- Pharmacophore Spatial Matching Results ---")
@printf("Total Matched Features: %d\n", length(matched_pairs))
@printf("Gaussian Overlay Fit Score: %.4f\n", fit_score)
println("Matched Pairs:")
for (q_idx, t_idx) in matched_pairs
    qf = ph_asp.features[q_idx]
    tf = ph_ibu.features[t_idx]
    dist = norm(qf.center .- tf.center)
    @printf("  Aspirin %-18s <---> Ibuprofen %-18s (Dist: %.2f Å)\n",
            qf.name, tf.name, dist)
end
println("===================================================================")
