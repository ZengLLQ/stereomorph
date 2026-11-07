# StereoMorph — Metrization & Trial Distance Sampling Module

module Metrization

using LinearAlgebra
using Random
using ..Types
using ..Bounds

export sample_trial_distances, crippen_havel_metrize, cayley_menger_det4, cayley_menger_det5

"""
    sample_trial_distances(bm::BoundsMatrix; rng::AbstractRNG=Random.GLOBAL_RNG)::Matrix{Float64}

Generates a symmetric trial distance matrix D where each distance d_ij is uniformly
sampled within [L_ij, U_ij].
"""
function sample_trial_distances(bm::BoundsMatrix; rng::AbstractRNG=Random.GLOBAL_RNG)::Matrix{Float64}
    n = bm.n_atoms
    D = zeros(Float64, n, n)
    for i in 1:n
        for j in (i + 1):n
            l = bm.lower[i, j]
            u = bm.upper[i, j]
            # Sample distance uniformly between lower and upper bounds
            d = l + rand(rng) * max(0.0, u - l)
            D[i, j] = D[j, i] = d
        end
    end
    return D
end

"""
    crippen_havel_metrize(bm::BoundsMatrix; rng::AbstractRNG=Random.GLOBAL_RNG)::Matrix{Float64}

Executes the Crippen-Havel incremental metrization algorithm:
Randomly selects atom pairs, assigns exact distances within current bounds,
and propagates triangle inequality adjustments dynamically to maintain metric consistency.
"""
function crippen_havel_metrize(bm::BoundsMatrix; rng::AbstractRNG=Random.GLOBAL_RNG)::Matrix{Float64}
    n = bm.n_atoms
    # Deep copy bounds matrix so original is unchanged
    L = copy(bm.lower)
    U = copy(bm.upper)
    D = zeros(Float64, n, n)

    # Collect all atom pairs
    pairs = Tuple{Int,Int}[]
    for i in 1:n
        for j in (i + 1):n
            push!(pairs, (i, j))
        end
    end

    # Shuffle pair order to sample random conformational space
    shuffle!(rng, pairs)

    for (i, j) in pairs
        l = L[i, j]
        u = U[i, j]

        # Choose exact distance
        d = (l >= u) ? l : (l + rand(rng) * (u - l))
        D[i, j] = D[j, i] = d

        # Fix bounds to exact distance
        L[i, j] = L[j, i] = d
        U[i, j] = U[j, i] = d

        # Propagate incremental triangle inequality to all other atoms k
        for k in 1:n
            if k == i || k == j continue end

            # Bounds for (i, k) using (j, k) and (i, j)
            u_ik_cand = d + U[j, k]
            if u_ik_cand < U[i, k]
                U[i, k] = U[k, i] = u_ik_cand
            end
            l_ik_cand1 = d - U[j, k]
            if l_ik_cand1 > L[i, k]
                L[i, k] = L[k, i] = l_ik_cand1
            end
            l_ik_cand2 = L[j, k] - d
            if l_ik_cand2 > L[i, k]
                L[i, k] = L[k, i] = l_ik_cand2
            end

            # Bounds for (j, k) using (i, k) and (i, j)
            u_jk_cand = d + U[i, k]
            if u_jk_cand < U[j, k]
                U[j, k] = U[k, j] = u_jk_cand
            end
            l_jk_cand1 = d - U[i, k]
            if l_jk_cand1 > L[j, k]
                L[j, k] = L[k, j] = l_jk_cand1
            end
            l_jk_cand2 = L[i, k] - d
            if l_jk_cand2 > L[j, k]
                L[j, k] = L[k, j] = l_jk_cand2
            end
        end
    end

    return D
end

"""
    cayley_menger_det4(D::Matrix{Float64}, p1::Int, p2::Int, p3::Int, p4::Int)::Float64

Calculates the 4-point Cayley-Menger determinant representing 288 * V^2 (volume of tetrahedron).
"""
function cayley_menger_det4(D::Matrix{Float64}, p1::Int, p2::Int, p3::Int, p4::Int)::Float64
    pts = [p1, p2, p3, p4]
    M = zeros(Float64, 5, 5)
    for i in 1:4
        for j in 1:4
            M[i, j] = D[pts[i], pts[j]]^2
        end
        M[i, 5] = 1.0
        M[5, i] = 1.0
    end
    M[5, 5] = 0.0
    return det(M)
end

"""
    cayley_menger_det5(D::Matrix{Float64}, p1::Int, p2::Int, p3::Int, p4::Int, p5::Int)::Float64

Calculates the 5-point Cayley-Menger determinant (must vanish if points embed in 3D).
"""
function cayley_menger_det5(D::Matrix{Float64}, p1::Int, p2::Int, p3::Int, p4::Int, p5::Int)::Float64
    pts = [p1, p2, p3, p4, p5]
    M = zeros(Float64, 6, 6)
    for i in 1:5
        for j in 1:5
            M[i, j] = D[pts[i], pts[j]]^2
        end
        M[i, 6] = 1.0
        M[6, i] = 1.0
    end
    M[6, 6] = 0.0
    return det(M)
end

end # module Metrization
