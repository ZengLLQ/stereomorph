# StereoMorph — Distance Bounds Matrix & Triangle Inequality Smoothing Module

module Bounds

using LinearAlgebra
using ..Types

export BoundsMatrix, compute_bounds, smooth_bounds!, check_bounds_consistency
export BoundsInconsistencyError

"""
    BoundsMatrix

Holds NxN lower and upper spatial distance bounds in Angstroms.
"""
mutable struct BoundsMatrix
    lower::Matrix{Float64}
    upper::Matrix{Float64}
    n_atoms::Int

    function BoundsMatrix(n::Int)
        lower = zeros(Float64, n, n)
        upper = fill(999.0, n, n)
        for i in 1:n
            lower[i, i] = 0.0
            upper[i, i] = 0.0
        end
        new(lower, upper, n)
    end
end

struct BoundsInconsistencyError <: Exception
    atom1::Int
    atom2::Int
    lower::Float64
    upper::Float64
    msg::String
end

function Base.showerror(io::IO, e::BoundsInconsistencyError)
    print(io, "BoundsInconsistencyError between atom $(e.atom1) and $(e.atom2): ")
    print(io, "lower bound ($(e.lower) Å) > upper bound ($(e.upper) Å). $(e.msg)")
end

"""
    compute_bounds(mol::Molecule)::BoundsMatrix

Constructs initial physical distance bounds based on covalent bonds (1-2),
geminal angles (1-3), dihedral conformations (1-4), and non-bonded contacts (1-5+).
"""
function compute_bounds(mol::Molecule)::BoundsMatrix
    n = length(mol.atoms)
    bm = BoundsMatrix(n)
    sp_matrix = shortest_path_matrix(mol)

    # 1. Direct covalent bonds (1-2 pairs)
    for b in mol.bonds
        i, j = b.atom1, b.atom2
        cov_i = mol.atoms[i].covalent_radius
        cov_j = mol.atoms[j].covalent_radius
        # Order correction: double/triple bonds are shorter
        order_corr = (b.order - 1.0) * 0.10
        ideal_d = cov_i + cov_j - order_corr
        tol = 0.04 # Tight bond length tolerance
        bm.lower[i, j] = bm.lower[j, i] = max(0.5, ideal_d - tol)
        bm.upper[i, j] = bm.upper[j, i] = ideal_d + tol
    end

    # 2. Geminal 1-3 pairs (atoms sharing a common neighbor k)
    for k in 1:n
        neighbors = mol.adj_list[k]
        for idx1 in 1:length(neighbors)
            for idx2 in (idx1 + 1):length(neighbors)
                i = neighbors[idx1]
                j = neighbors[idx2]

                # If already bonded (e.g. 3-membered ring), skip
                if sp_matrix[i, j] <= 1
                    continue
                end

                # Determine ideal bond angle theta around atom k
                hyb = mol.atoms[k].hybridization
                theta = if hyb == SP3
                    1.9106 # 109.47 degrees (tetrahedral)
                elseif hyb == SP2
                    2.0944 # 120.0 degrees (trigonal planar)
                elseif hyb == SP
                    3.14159 # 180.0 degrees (linear)
                else
                    1.9106
                end

                r_ik = 0.5 * (bm.lower[i, k] + bm.upper[i, k])
                r_jk = 0.5 * (bm.lower[j, k] + bm.upper[j, k])

                # Law of cosines: d_ij^2 = r_ik^2 + r_jk^2 - 2 r_ik r_jk cos(theta)
                d13_ideal = sqrt(max(0.1, r_ik^2 + r_jk^2 - 2.0 * r_ik * r_jk * cos(theta)))
                tol_angle = 0.08

                bm.lower[i, j] = bm.lower[j, i] = max(bm.lower[i, j], d13_ideal - tol_angle)
                bm.upper[i, j] = bm.upper[j, i] = min(bm.upper[i, j], d13_ideal + tol_angle)
            end
        end
    end

    # 3. Dihedral 1-4 pairs (connected via a 3-bond path i - a - b - j)
    for b in mol.bonds
        a, b_atom = b.atom1, b.atom2
        for i in mol.adj_list[a]
            if i == b_atom continue end
            for j in mol.adj_list[b_atom]
                if j == a || j == i continue end
                if sp_matrix[i, j] != 3 continue end

                r_ia = 0.5 * (bm.lower[i, a] + bm.upper[i, a])
                r_ab = 0.5 * (bm.lower[a, b_atom] + bm.upper[a, b_atom])
                r_bj = 0.5 * (bm.lower[b_atom, j] + bm.upper[b_atom, j])

                # Approximate cis distance (dihedral phi = 0) and trans distance (phi = 180)
                # For standard tetrahedral/trigonal angles:
                theta_a = (mol.atoms[a].hybridization == SP2) ? 2.0944 : 1.9106
                theta_b = (mol.atoms[b_atom].hybridization == SP2) ? 2.0944 : 1.9106

                # Coordinate projection:
                # trans: d^2 ≈ (r_ia sin θ_a + r_bj sin θ_b)^2 + (r_ab - r_ia cos θ_a - r_bj cos θ_b)^2
                # cis:   d^2 ≈ (r_ia sin θ_a - r_bj sin θ_b)^2 + (r_ab - r_ia cos θ_a - r_bj cos θ_b)^2
                x_trans = r_ab - r_ia * cos(theta_a) - r_bj * cos(theta_b)
                y_trans = r_ia * sin(theta_a) + r_bj * sin(theta_b)
                d_trans = sqrt(x_trans^2 + y_trans^2)

                x_cis = r_ab - r_ia * cos(theta_a) - r_bj * cos(theta_b)
                y_cis = r_ia * sin(theta_a) - r_bj * sin(theta_b)
                d_cis = sqrt(x_cis^2 + y_cis^2)

                # Check if restricted by planar/aromatic ring
                is_rigid = b.aromatic || b.order >= 2.0
                if is_rigid
                    # Restricted to trans or cis
                    bm.lower[i, j] = bm.lower[j, i] = max(bm.lower[i, j], d_cis - 0.15)
                    bm.upper[i, j] = bm.upper[j, i] = min(bm.upper[i, j], d_trans + 0.15)
                else
                    vdw_contact = mol.atoms[i].vdw_radius + mol.atoms[j].vdw_radius - 0.4
                    bm.lower[i, j] = bm.lower[j, i] = max(bm.lower[i, j], max(vdw_contact, d_cis * 0.85))
                    bm.upper[i, j] = bm.upper[j, i] = min(bm.upper[i, j], d_trans + 0.20)
                end
            end
        end
    end

    # 4. Long-range / non-bonded 1-5+ pairs
    for i in 1:n
        for j in (i + 1):n
            if sp_matrix[i, j] >= 4
                vdw_sum = mol.atoms[i].vdw_radius + mol.atoms[j].vdw_radius - 0.35
                bm.lower[i, j] = bm.lower[j, i] = max(bm.lower[i, j], vdw_sum)

                # Upper bound is the sum of bond upper bounds along the shortest path
                # Estimated as graph_distance * average_bond_length * 1.3
                path_len = sp_matrix[i, j]
                bm.upper[i, j] = bm.upper[j, i] = min(bm.upper[i, j], path_len * 1.55 * 1.25)
            end
        end
    end

    # Apply triangle inequality smoothing
    smooth_bounds!(bm)
    check_bounds_consistency(bm)

    return bm
end

"""
    smooth_bounds!(bm::BoundsMatrix)

Applies Floyd-Warshall all-pairs triangle inequality smoothing:
- Upper bound: U_ij <= U_ik + U_kj
- Lower bound: L_ij >= L_ik - U_kj and L_ij >= L_jk - U_ki
"""
function smooth_bounds!(bm::BoundsMatrix)
    n = bm.n_atoms
    U = bm.upper
    L = bm.lower

    # Floyd-Warshall metric relaxation: O(N^3)
    for k in 1:n
        for i in 1:n
            if i == k continue end
            u_ik = U[i, k]
            l_ik = L[i, k]

            for j in 1:n
                if j == i || j == k continue end
                u_kj = U[k, j]
                l_kj = L[k, j]

                # Upper bound triangle rule
                if u_ik + u_kj < U[i, j]
                    U[i, j] = u_ik + u_kj
                end

                # Lower bound triangle rule
                cand_lower1 = l_ik - u_kj
                if cand_lower1 > L[i, j]
                    L[i, j] = cand_lower1
                end

                cand_lower2 = l_kj - u_ik
                if cand_lower2 > L[i, j]
                    L[i, j] = cand_lower2
                end
            end
        end
    end

    # Ensure symmetry and self-distances
    for i in 1:n
        L[i, i] = 0.0
        U[i, i] = 0.0
        for j in (i + 1):n
            u_sym = min(U[i, j], U[j, i])
            l_sym = max(L[i, j], L[j, i])
            U[i, j] = U[j, i] = u_sym
            L[i, j] = L[j, i] = l_sym
        end
    end
end

"""
    check_bounds_consistency(bm::BoundsMatrix)

Verifies that L_ij <= U_ij for all i, j. Throws BoundsInconsistencyError if violated.
"""
function check_bounds_consistency(bm::BoundsMatrix)
    n = bm.n_atoms
    tol = 1e-5
    for i in 1:n
        for j in (i + 1):n
            if bm.lower[i, j] > bm.upper[i, j] + tol
                throw(BoundsInconsistencyError(
                    i, j, bm.lower[i, j], bm.upper[i, j],
                    "Triangle inequality smoothing revealed conflicting topological distance constraints."
                ))
            end
        end
    end
    return true
end

end # module Bounds
