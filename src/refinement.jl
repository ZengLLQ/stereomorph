# StereoMorph — Force Field & Distance Geometry Penalty Refinement Module

module Refinement

using LinearAlgebra
using Statistics: mean
using ..Types
using ..Bounds
using ..Embedding

export refine_conformation!, evaluate_energy_and_gradient, count_violations

"""
    evaluate_energy_and_gradient(coords::Matrix{Float64}, mol::Molecule, bm::BoundsMatrix;
                                 w_bounds::Float64=1.0,
                                 w_chiral::Float64=10.0,
                                 w_planar::Float64=5.0,
                                 w_steric::Float64=2.0)

Computes the total distance geometry penalty energy and its analytical gradient (3 x N).
"""
function evaluate_energy_and_gradient(coords::Matrix{Float64}, mol::Molecule, bm::BoundsMatrix;
                                     w_bounds::Float64=1.0,
                                     w_chiral::Float64=10.0,
                                     w_planar::Float64=5.0,
                                     w_steric::Float64=2.0)::Tuple{Float64, Matrix{Float64}}
    n = length(mol.atoms)
    E_total = 0.0
    grad = zeros(Float64, 3, n)

    U = bm.upper
    L = bm.lower

    # 1. Distance Bounds Penalty
    for i in 1:n
        for j in (i + 1):n
            u2 = U[i, j]^2
            l2 = L[i, j]^2
            if u2 < 1e-6 || l2 < 1e-6 continue end

            # Vector r_ij = r_i - r_j
            dx = coords[1, i] - coords[1, j]
            dy = coords[2, i] - coords[2, j]
            dz = coords[3, i] - coords[3, j]
            d2 = dx^2 + dy^2 + dz^2

            # Upper bound violation: d^2 > U^2
            if d2 > u2
                delta = (d2 - u2) / u2
                E_total += w_bounds * delta^2
                # dE/d(d2) = 2 * w_bounds * delta * (1 / u2)
                dE_dd2 = 2.0 * w_bounds * delta / u2
                gx = 2.0 * dE_dd2 * dx
                gy = 2.0 * dE_dd2 * dy
                gz = 2.0 * dE_dd2 * dz

                grad[1, i] += gx; grad[2, i] += gy; grad[3, i] += gz
                grad[1, j] -= gx; grad[2, j] -= gy; grad[3, j] -= gz
            elseif d2 < l2
                # Lower bound violation: d^2 < L^2
                delta = (l2 - d2) / l2
                E_total += w_bounds * delta^2
                # dE/d(d2) = -2 * w_bounds * delta / l2
                dE_dd2 = -2.0 * w_bounds * delta / l2
                gx = 2.0 * dE_dd2 * dx
                gy = 2.0 * dE_dd2 * dy
                gz = 2.0 * dE_dd2 * dz

                grad[1, i] += gx; grad[2, i] += gy; grad[3, i] += gz
                grad[1, j] -= gx; grad[2, j] -= gy; grad[3, j] -= gz
            end
        end
    end

    # 2. Chiral Centers Stereochemical Restraints
    V_min = 1.2 # Ideal minimum signed tetrahedral volume in Å^3
    for cc in mol.chiral_centers
        c = cc.center
        n1, n2, n3 = cc.neighbors[1], cc.neighbors[2], cc.neighbors[3]
        target = cc.target_sign

        r_c = coords[:, c]
        u = coords[:, n1] .- r_c
        v = coords[:, n2] .- r_c
        w = coords[:, n3] .- r_c

        # Vector cross products
        v_cross_w = [v[2]*w[3] - v[3]*w[2], v[3]*w[1] - v[1]*w[3], v[1]*w[2] - v[2]*w[1]]
        w_cross_u = [w[2]*u[3] - w[3]*u[2], w[3]*u[1] - w[1]*u[3], w[1]*u[2] - w[2]*u[1]]
        u_cross_v = [u[2]*v[3] - u[3]*v[2], u[3]*v[1] - u[1]*v[3], u[1]*v[2] - u[2]*v[1]]

        V_curr = u[1]*v_cross_w[1] + u[2]*v_cross_w[2] + u[3]*v_cross_w[3]

        # Desired signed volume is target * V_min
        # Penalty if target * V_curr < V_min
        vol_defect = V_min - (target * V_curr)
        if vol_defect > 0.0
            E_total += w_chiral * vol_defect^2

            # Analytical gradient of V with respect to coordinates
            # dE/dV = -2 * w_chiral * vol_defect * target
            dE_dV = -2.0 * w_chiral * vol_defect * target

            g_n1 = dE_dV .* v_cross_w
            g_n2 = dE_dV .* w_cross_u
            g_n3 = dE_dV .* u_cross_v
            g_c  = -(g_n1 .+ g_n2 .+ g_n3)

            grad[:, n1] .+= g_n1
            grad[:, n2] .+= g_n2
            grad[:, n3] .+= g_n3
            grad[:, c]  .+= g_c
        end
    end

    # 3. Planar Systems (Aromatic Rings & Amides via Exact Improper Volume)
    for ps in mol.planar_systems
        atoms = ps.atoms
        if length(atoms) < 4 continue end

        a1, a2, a3 = atoms[1], atoms[2], atoms[3]
        r1 = coords[:, a1]
        r2 = coords[:, a2]
        r3 = coords[:, a3]

        u = r2 .- r1
        v = r3 .- r1

        # Precompute u x v
        u_cross_v = [u[2]*v[3] - u[3]*v[2], u[3]*v[1] - u[1]*v[3], u[1]*v[2] - u[2]*v[1]]

        for idx in 4:length(atoms)
            al = atoms[idx]
            rl = coords[:, al]
            w = rl .- r1

            # Signed tetrahedral volume V = w ⋅ (u × v)
            V = w[1]*u_cross_v[1] + w[2]*u_cross_v[2] + w[3]*u_cross_v[3]

            E_total += w_planar * V^2

            # Analytical gradient of V^2: 2 * w_planar * V * dV/dr
            factor = 2.0 * w_planar * V

            # dV/drl = u x v
            g_l = factor .* u_cross_v

            # dV/dr3 = w x u
            w_cross_u = [w[2]*u[3] - w[3]*u[2], w[3]*u[1] - w[1]*u[3], w[1]*u[2] - w[2]*u[1]]
            g_3 = factor .* w_cross_u

            # dV/dr2 = v x w
            v_cross_w = [v[2]*w[3] - v[3]*w[2], v[3]*w[1] - v[1]*w[3], v[1]*w[2] - v[2]*w[1]]
            g_2 = factor .* v_cross_w

            # dV/dr1 = -(g_l + g_3 + g_2)
            g_1 = -(g_l .+ g_3 .+ g_2)

            grad[:, al] .+= g_l
            grad[:, a3] .+= g_3
            grad[:, a2] .+= g_2
            grad[:, a1] .+= g_1
        end
    end

    # 4. Steric Non-Bonded Clashes
    sp_matrix = shortest_path_matrix(mol)
    for i in 1:n
        for j in (i + 1):n
            if sp_matrix[i, j] >= 4
                contact_d = mol.atoms[i].vdw_radius + mol.atoms[j].vdw_radius - 0.35
                contact_d2 = contact_d^2

                dx = coords[1, i] - coords[1, j]
                dy = coords[2, i] - coords[2, j]
                dz = coords[3, i] - coords[3, j]
                d2 = dx^2 + dy^2 + dz^2

                if d2 < contact_d2
                    delta = (contact_d2 - d2) / contact_d2
                    E_total += w_steric * delta^2

                    dE_dd2 = -2.0 * w_steric * delta / contact_d2
                    gx = 2.0 * dE_dd2 * dx
                    gy = 2.0 * dE_dd2 * dy
                    gz = 2.0 * dE_dd2 * dz

                    grad[1, i] += gx; grad[2, i] += gy; grad[3, i] += gz
                    grad[1, j] -= gx; grad[2, j] -= gy; grad[3, j] -= gz
                end
            end
        end
    end

    return E_total, grad
end

"""
    count_violations(coords::Matrix{Float64}, mol::Molecule, bm::BoundsMatrix; tol::Float64=0.08)::Int

Counts number of distance bounds violations that exceed the specified tolerance.
"""
function count_violations(coords::Matrix{Float64}, mol::Molecule, bm::BoundsMatrix; tol::Float64=0.08)::Int
    n = length(mol.atoms)
    violations = 0
    for i in 1:n
        for j in (i + 1):n
            u = bm.upper[i, j]
            l = bm.lower[i, j]
            if u < 1e-6 continue end

            dx = coords[1, i] - coords[1, j]
            dy = coords[2, i] - coords[2, j]
            dz = coords[3, i] - coords[3, j]
            d = sqrt(dx^2 + dy^2 + dz^2)

            if d > (u + tol) || d < (l - tol)
                violations += 1
            end
        end
    end
    return violations
end

"""
    refine_conformation!(coords::Matrix{Float64}, mol::Molecule, bm::BoundsMatrix;
                         max_iter::Int=300,
                         tol_grad::Float64=1e-4,
                         method::Symbol=:lbfgs)::Tuple{Float64, Bool, Int}

Refines molecular 3D coordinates using L-BFGS or Polak-Ribière Conjugate Gradient optimization
with Armijo line search.
Returns (final_energy, converged, iterations_performed).
"""
function refine_conformation!(coords::Matrix{Float64}, mol::Molecule, bm::BoundsMatrix;
                             max_iter::Int=300,
                             tol_grad::Float64=1e-4,
                             method::Symbol=:lbfgs)::Tuple{Float64, Bool, Int}
    n = length(mol.atoms)
    dim = 3 * n

    # Flatten coordinates to 1D vector x
    x = vec(coords)

    function obj_and_grad(x_vec::Vector{Float64})
        c_mat = reshape(x_vec, 3, n)
        e, g_mat = evaluate_energy_and_gradient(c_mat, mol, bm)
        return e, vec(g_mat)
    end

    energy, grad = obj_and_grad(x)
    if norm(grad, Inf) < tol_grad
        return energy, true, 0
    end

    converged = false
    iter = 0

    if method == :lbfgs
        # L-BFGS implementation with history memory m = 7
        m = 7
        s_history = Vector{Vector{Float64}}()
        y_history = Vector{Vector{Float64}}()
        rho_history = Float64[]

        while iter < max_iter
            iter += 1

            # L-BFGS two-loop recursion for search direction p = -H_k * grad
            q = copy(grad)
            k_hist = length(s_history)
            alpha_hist = zeros(Float64, k_hist)

            for i in k_hist:-1:1
                alpha_hist[i] = rho_history[i] * dot(s_history[i], q)
                q .-= alpha_hist[i] .* y_history[i]
            end

            # Initial Hessian approximation H_0 = gamma * I
            gamma = 1.0
            if k_hist > 0
                s_last = s_history[end]
                y_last = y_history[end]
                y_dot_y = dot(y_last, y_last)
                if y_dot_y > 1e-12
                    gamma = dot(s_last, y_last) / y_dot_y
                end
            end
            r = gamma .* q

            for i in 1:k_hist
                beta = rho_history[i] * dot(y_history[i], r)
                r .+= s_history[i] .* (alpha_hist[i] - beta)
            end

            dir = -r
            # Ensure descent direction
            if dot(dir, grad) > -1e-6 * norm(dir) * norm(grad)
                dir = -grad
            end

            # Armijo backtracking line search
            step = 1.0
            c1 = 1e-4
            dir_dot_grad = dot(dir, grad)
            new_x = x .+ step .* dir
            new_energy, new_grad = obj_and_grad(new_x)

            line_iter = 0
            while new_energy > (energy + c1 * step * dir_dot_grad) && line_iter < 20
                step *= 0.5
                new_x = x .+ step .* dir
                new_energy, new_grad = obj_and_grad(new_x)
                line_iter += 1
            end

            # Check convergence
            s = new_x .- x
            y = new_grad .- grad

            x .= new_x
            energy = new_energy
            grad .= new_grad

            if norm(grad, Inf) < tol_grad || norm(s) < 1e-6
                converged = true
                break
            end

            # Update L-BFGS history
            s_dot_y = dot(s, y)
            if s_dot_y > 1e-10
                if length(s_history) >= m
                    popfirst!(s_history)
                    popfirst!(y_history)
                    popfirst!(rho_history)
                end
                push!(s_history, s)
                push!(y_history, y)
                push!(rho_history, 1.0 / s_dot_y)
            end
        end
    else
        # Polak-Ribiere Conjugate Gradient
        dir = -copy(grad)
        while iter < max_iter
            iter += 1
            step = 1.0
            c1 = 1e-4
            dir_dot_grad = dot(dir, grad)
            if dir_dot_grad >= 0.0
                dir = -grad
                dir_dot_grad = dot(dir, grad)
            end

            new_x = x .+ step .* dir
            new_energy, new_grad = obj_and_grad(new_x)

            line_iter = 0
            while new_energy > (energy + c1 * step * dir_dot_grad) && line_iter < 20
                step *= 0.5
                new_x = x .+ step .* dir
                new_energy, new_grad = obj_and_grad(new_x)
                line_iter += 1
            end

            if norm(new_grad, Inf) < tol_grad
                x .= new_x
                energy = new_energy
                converged = true
                break
            end

            # Polak-Ribiere update parameter beta
            y = new_grad .- grad
            beta_pr = dot(new_grad, y) / max(1e-12, dot(grad, grad))
            beta = max(0.0, beta_pr)

            dir = -new_grad .+ beta .* dir
            x .= new_x
            energy = new_energy
            grad .= new_grad
        end
    end

    # Reshape back to coords matrix
    coords .= reshape(x, 3, n)
    return energy, converged, iter
end

end # module Refinement
