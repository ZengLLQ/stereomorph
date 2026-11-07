# StereoMorph — Conformation Superposition, RMSD & Kabsch Alignment Module

module Alignment

using LinearAlgebra
using Statistics: mean
using ..Types

export center_coordinates, kabsch_rotation, align_coordinates, calculate_rmsd
export pairwise_rmsd_matrix, cluster_conformers

"""
    center_coordinates(coords::Matrix{Float64})::Tuple{Matrix{Float64}, Vector{Float64}}

Translates 3D coordinates so their geometric centroid is at the origin (0, 0, 0).
Returns (centered_coords, centroid_vector).
"""
function center_coordinates(coords::Matrix{Float64})::Tuple{Matrix{Float64}, Vector{Float64}}
    centroid = [mean(coords[1, :]), mean(coords[2, :]), mean(coords[3, :])]
    centered = coords .- centroid
    return centered, centroid
end

"""
    kabsch_rotation(P::Matrix{Float64}, Q::Matrix{Float64})::Matrix{Float64}

Computes the optimal 3x3 proper rotation matrix R (with det(R) = +1) that minimizes
the RMSD between coordinate sets P and Q using the Kabsch SVD algorithm.
"""
function kabsch_rotation(P::Matrix{Float64}, Q::Matrix{Float64})::Matrix{Float64}
    # 1. Center both coordinate sets
    P_cent, _ = center_coordinates(P)
    Q_cent, _ = center_coordinates(Q)

    # 2. Compute 3x3 covariance matrix H = P_cent * Q_cent^T
    H = P_cent * Q_cent'

    # 3. Singular Value Decomposition H = U * S * V^T
    F = svd(H)
    U = F.U
    V = F.V

    # 4. Check for proper rotation (det(R) must be +1, not -1 reflection)
    d = det(V * U')
    S_corr = Matrix{Float64}(I, 3, 3)
    S_corr[3, 3] = (d < 0.0) ? -1.0 : 1.0

    # 5. Optimal rotation matrix R = V * S_corr * U^T
    R = V * S_corr * U'
    return R
end

"""
    align_coordinates(P::Matrix{Float64}, Q::Matrix{Float64})::Tuple{Matrix{Float64}, Matrix{Float64}, Float64}

Rotates and translates P to best superimpose onto Q.
Returns (aligned_P, rotation_matrix, rmsd_value).
"""
function align_coordinates(P::Matrix{Float64}, Q::Matrix{Float64})::Tuple{Matrix{Float64}, Matrix{Float64}, Float64}
    P_cent, p_centroid = center_coordinates(P)
    Q_cent, q_centroid = center_coordinates(Q)

    R = kabsch_rotation(P, Q)
    # Apply rotation and translate to Q's centroid
    P_aligned = (R * P_cent) .+ q_centroid

    # Compute RMSD
    diff = P_aligned .- Q
    n = size(P, 2)
    rmsd_val = sqrt(sum(diff .^ 2) / n)

    return P_aligned, R, rmsd_val
end

"""
    calculate_rmsd(P::Matrix{Float64}, Q::Matrix{Float64})::Float64

Calculates the optimal Kabsch-aligned Root-Mean-Square Deviation (RMSD) between P and Q.
"""
function calculate_rmsd(P::Matrix{Float64}, Q::Matrix{Float64})::Float64
    _, _, rmsd_val = align_coordinates(P, Q)
    return rmsd_val
end

"""
    pairwise_rmsd_matrix(conformers::Vector{Conformation})::Matrix{Float64}

Computes the symmetric pairwise RMSD distance matrix across an ensemble of conformations.
"""
function pairwise_rmsd_matrix(conformers::Vector{Conformation})::Matrix{Float64}
    k = length(conformers)
    mat = zeros(Float64, k, k)
    for i in 1:k
        for j in (i + 1):k
            val = calculate_rmsd(conformers[i].coords, conformers[j].coords)
            mat[i, j] = mat[j, i] = val
        end
    end
    return mat
end

"""
    cluster_conformers(conformers::Vector{Conformation}; cutoff::Float64=0.6)::Tuple{Vector{Conformation}, Vector{Int}}

Performs greedy leader clustering on a conformer ensemble using an RMSD threshold cutoff.
Conformers are sorted by energy, and the lowest energy conformer initializes each cluster.
Returns (cluster_representatives, cluster_assignment_indices).
"""
function cluster_conformers(conformers::Vector{Conformation}; cutoff::Float64=0.6)::Tuple{Vector{Conformation}, Vector{Int}}
    k = length(conformers)
    if k == 0
        return Conformation[], Int[]
    end

    # Sort conformers by ascending energy
    order = sortperm([c.energy for c in conformers])
    sorted_confs = conformers[order]

    representatives = Conformation[]
    assignments = zeros(Int, k)

    for i in 1:k
        conf = sorted_confs[i]
        matched_cluster = 0

        for (c_idx, rep) in enumerate(representatives)
            dist = calculate_rmsd(conf.coords, rep.coords)
            if dist < cutoff
                matched_cluster = c_idx
                break
            end
        end

        if matched_cluster == 0
            push!(representatives, conf)
            assignments[order[i]] = length(representatives)
        else
            assignments[order[i]] = matched_cluster
        end
    end

    return representatives, assignments
end

end # module Alignment
