# StereoMorph — Metric Matrix Construction & Classical MDS Embedding Module

module Embedding

using LinearAlgebra
using ..Types

export embed_distance_matrix, orient_chirality!, signed_chiral_volume

"""
    embed_distance_matrix(D::Matrix{Float64})::Tuple{Matrix{Float64}, Float64}

Performs Classical Multidimensional Scaling (MDS / Principal Coordinate Analysis)
on a trial distance matrix D to construct initial 3D Cartesian coordinates (3 x N).
Returns (coords, embedding_quality).
"""
function embed_distance_matrix(D::Matrix{Float64})::Tuple{Matrix{Float64}, Float64}
    n = size(D, 1)

    # 1. Compute squared distance matrix D^(2)
    D2 = D .^ 2

    # 2. Centering matrix H = I - (1/n) * 1 * 1^T
    H = Matrix{Float64}(I, n, n) .- (1.0 / n)

    # 3. Metric (Gram) matrix G = -0.5 * H * D2 * H
    G = -0.5 .* (H * D2 * H)
    # Ensure exact numerical symmetry
    G = 0.5 .* (G .+ G')

    # 4. Eigendecomposition of symmetric matrix G
    F = eigen(Symmetric(G))
    vals = F.values
    vecs = F.vectors

    # Sort eigenvalues in descending order
    p = sortperm(vals, rev=true)
    sorted_vals = vals[p]
    sorted_vecs = vecs[:, p]

    # Calculate 3D embedding quality: ratio of top 3 positive eigenvalues to total absolute eigenvalues
    pos_top3 = sum(max(0.0, sorted_vals[k]) for k in 1:min(3, n))
    tot_abs = sum(abs.(sorted_vals))
    quality = (tot_abs > 1e-12) ? (pos_top3 / tot_abs) : 1.0

    # 5. Extract 3D coordinates: X = diag(sqrt(lambda_1, lambda_2, lambda_3)) * V_{1:3}^T
    coords = zeros(Float64, 3, n)
    for dim in 1:min(3, n)
        val = sorted_vals[dim]
        factor = (val > 0.0) ? sqrt(val) : 0.0
        coords[dim, :] .= factor .* sorted_vecs[:, dim]
    end

    # If n < 3, remaining dimensions are zero
    return coords, quality
end

"""
    signed_chiral_volume(coords::Matrix{Float64}, center::Int, n1::Int, n2::Int, n3::Int)::Float64

Computes the signed scalar triple product volume of three neighbor vectors relative to a chiral center:
V = (r_n1 - r_c) ⋅ ((r_n2 - r_c) × (r_n3 - r_c))
"""
function signed_chiral_volume(coords::Matrix{Float64}, center::Int, n1::Int, n2::Int, n3::Int)::Float64
    r_c = coords[:, center]
    u = coords[:, n1] .- r_c
    v = coords[:, n2] .- r_c
    w = coords[:, n3] .- r_c

    # Cross product v × w
    cross_vw = [
        v[2] * w[3] - v[3] * w[2],
        v[3] * w[1] - v[1] * w[3],
        v[1] * w[2] - v[2] * w[1]
    ]

    return u[1] * cross_vw[1] + u[2] * cross_vw[2] + u[3] * cross_vw[3]
end

"""
    orient_chirality!(coords::Matrix{Float64}, mol::Molecule)::Bool

Inspects all chiral centers in the molecule. If the majority of centers are inverted relative
to their specified target stereochemistry, reflects the entire coordinate system across the z-axis.
Returns true if reflection was performed, false otherwise.
"""
function orient_chirality!(coords::Matrix{Float64}, mol::Molecule)::Bool
    if isempty(mol.chiral_centers)
        return false
    end

    inversions = 0
    total_centers = length(mol.chiral_centers)

    for cc in mol.chiral_centers
        c = cc.center
        n1, n2, n3 = cc.neighbors[1], cc.neighbors[2], cc.neighbors[3]
        v_current = signed_chiral_volume(coords, c, n1, n2, n3)

        # Check if sign matches cc.target_sign
        if (v_current * cc.target_sign) < 0.0
            inversions += 1
        end
    end

    # If more than half are inverted, reflect across z-axis
    if inversions > total_centers / 2
        coords[3, :] .= -coords[3, :]
        return true
    end

    return false
end

end # module Embedding
