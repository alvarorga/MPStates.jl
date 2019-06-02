#
# DMRG algorithms.
#

"""
    dmrg_1!(H::Mpo{T}, psi::Mps{T}, D::Int;
            max_iters::Int=500, tol::Float64=1e-6,
            debug::Int=1) where T<:Number

Find the ground state of `H` with bond dimension `D` starting with the initial
solution `psi0`. Use the basic 1-site DMRG algorithm without any sophistication.
"""
function dmrg_1!(H::Mpo{T}, psi::Mps{T}, D::Int;
                 max_iters::Int=500, tol::Float64=1e-6,
                 debug::Int=1) where T<:Number
    # Increase bond dimension.
    current_D = ceil(Int, D/5)
    enlarge_bond_dimension!(psi, current_D)

    # Create left and right environments.
    L_env = fill(ones(T, 1, 1, 1), psi.L)
    R_env = fill(ones(T, 1, 1, 1), psi.L)
    # Initialize left environment.
    for i=2:psi.L
        L_env[i] = prop_right3(L_env[i-1], psi.A[i-1], H.M[i-1], psi.A[i-1])
    end

    # Compute energy of state and variance at each sweep.
    E = zeros(max_iters+1)
    E[1] = 1e5
    var = zeros(max_iters+1)
    var[1] = 1e5
    var_iter = 1e5
    # Keep number of iterations done.
    it = 2
    var_has_converged = false
    while var_iter > tol && it <= max_iters+1 && !var_has_converged
        # Do complete left and right sweeps.
        do_sweep!(psi, H, L_env, R_env, -1, debug)
        do_sweep!(psi, H, L_env, R_env, +1, debug)

        # Compute energy and variance of `psi` after sweeps.
        E[it] = expected(H, psi)
        var[it] = m_variance(H, psi)
        var_has_converged = abs(var[it] - var[it-1]) < 1e-8
        if debug > 0
            println("Done sweep $it")
            println("    E: ", @sprintf("%.2e", E[it]),
                    ", ΔE: ", @sprintf("%.2e", E[it]-E[it-1]))
            println("    var: ", @sprintf("%.2e", var[it]),
                    ", Δvar: ", @sprintf("%.2e", var[it]-var[it-1]))
        end

        # If variance converges enlarge bond dimension until it reaches `max_D`.
        if abs(var[it] - var[it-1])/var[it] < 1e-3 && current_D != D
            current_D = min(current_D + ceil(Int, D/5), D)
            enlarge_bond_dimension!(psi, current_D)
            if debug > 0
                println("    D: $(current_D)")
            end
        end

        # Update loop control parameters.
        var_iter = var[it]
        it += 1
    end
    return E[1:it-1], var[1:it-1]
end

"""
    update_lr_envs!(i::Int, Ai::Array{T, 2}, psi::Mps{T}, H::Mpo{T},
                    Le::Vector{Array{T, 3}}, Re::Vector{Array{T, 3}},
                    sense::Int) where T<:Number

Update the left and right environments after the local Hamiltonian is minimized.
"""
function update_lr_envs!(i::Int, Ai::Array{T, 2}, psi::Mps{T}, H::Mpo{T},
                         Le::Vector{Array{T, 3}}, Re::Vector{Array{T, 3}},
                         sense::Int) where T<:Number
    if sense == +1
        Ai = reshape(Ai, (size(Le[i], 1)*psi.d, size(Re[i], 1)))
        Qa, Ra = qr(Ai)
        Qa = Matrix(Qa)
        Qa = reshape(Qa, (size(Le[i], 1), psi.d, size(Qa, 2)))

        # 3. Update left and right environments at L[i+1] and R[i] and psi.
        psi.A[i] = Qa
        if i < psi.L
            Le[i+1] = MPStates.prop_right3(Le[i], Qa, H.M[i], Qa)
            @tensor Re[i][r1, r2, r3] = Ra[r1, a]*Ra[r3, b]*Re[i][a, r2, b]
        end
    else
        Ai = reshape(Ai, (size(Le[i], 1), psi.d*size(Re[i], 1)))
        La, Qa = lq(Ai)
        Qa = Matrix(Qa)
        Qa = reshape(Qa, (size(Qa, 1), psi.d, size(Re[i], 1)))

        # 3. Update left and right environments at L[i] and R[i-1] and psi.
        psi.A[i] = Qa
        if i > 1
            Re[i-1] = MPStates.prop_left3(Qa, H.M[i], Qa, Re[i])
            @tensor Le[i][l1, l2, l3] = Le[i][a, l2, b]*La[a, l1]*La[b, l3]
        end
    end
    return
end

"""
    do_sweep!(psi::Mps{T}, H::Mpo{T},
              Le::Vector{Array{T, 3}}, Re::Vector{Array{T, 3}},
              sense::Int, debug::Int=0) where T<:Number

Do a sweep to locally minimize the energy of `psi`. The direction of the sweep
is given by `sense = +1, -1`.
"""
function do_sweep!(psi::Mps{T}, H::Mpo{T},
                   Le::Vector{Array{T, 3}}, Re::Vector{Array{T, 3}},
                   sense::Int, debug::Int=0) where T<:Number

    sense == 1 || sense == -1 || throw("`Sense` must be either `+1` or `-1`.")
    # Order of sites to do the sweep.
    sweep_sites = sense == +1 ? (1:psi.L) : reverse(1:psi.L)
    for i in sweep_sites
        # 1. Compute local Hamiltonian.
        @tensor Hi[l1, s1, r1, l3, s2, r3] := (Le[i][l1, l2, l3]
                                               *H.M[i][l2, s1, s2, r2]
                                               *Re[i][r1, r2, r3])
        Hi = reshape(Hi, (size(Le[i], 1)*psi.d*size(Re[i], 1),
                          size(Le[i], 3)*psi.d*size(Re[i], 3)))

        # 2. Compute lowest eigenmode of Hi.
        local_E, Ai = eigen(Hermitian(Hi), 1:1)

        # 3. Update left and right environments at L[i+1] and R[i], and psi.
        update_lr_envs!(i, Ai, psi, H, Le, Re, sense)

        # 4. Print info.
        debug > 1 && println("iter: $i, norm(Hi): $(round(norm(Hi), digits=3))")
    end
    return
end
