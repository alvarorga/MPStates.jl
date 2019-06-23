#
# DMRG algorithms.
#

"""
    minimize!(H::Mpo{T}, psi::Mps{T}, D::Int, algorithm::String="DMRG1";
              max_iters::Int=500, tol::Float64=1e-6,
              debug::Int=1) where T<:Number

Minimize energy of `psi` with respect to `H`, allowing maximum bond dimension
`D`.

Return the energy and variance of `psi` at every sweep. The `debug` parameter
controls the amount of information the script outputs at runtime. Possible
algorithms to minimize the energy are: "DMRG1": 1-site DMRG.
"""
function minimize!(H::Mpo{T}, psi::Mps{T}, D::Int, algorithm::String="DMRG1";
                   max_iters::Int=500, tol::Float64=1e-6,
                   debug::Int=1) where T<:Number
    # Increase bond dimension.
    current_D = ceil(Int, D/5)
    enlarge_bond_dimension!(psi, current_D)

    # Create left and right environments.
    Le = fill(ones(T, 1, 1, 1), psi.L)
    Re = fill(ones(T, 1, 1, 1), psi.L)
    # Initialize left environment.
    for i=2:psi.L
        Le[i] = prop_right3(Le[i-1], psi.A[i-1], H.M[i-1], psi.A[i-1])
    end

    # Compute energy of state and variance at each sweep. We start at iteration
    # `it` = 2, starting with artificially big values in the first iteration.
    E = [1e5]
    var = [1e5]
    it = 2
    var_is_stuck = false
    while var[it-1] > tol && it <= max_iters+1 && !var_is_stuck
        # Do left and right sweeps.
        if algorithm == "DMRG1"
            do_sweep_1s!(psi, H, Le, Re, -1, debug)
            do_sweep_1s!(psi, H, Le, Re, +1, debug)
        else
            do_sweep_2s!(psi, H, Le, Re, -1, current_D, debug)
            do_sweep_2s!(psi, H, Le, Re, +1, current_D, debug)
        end

        # Compute energy and variance of `psi` after sweeps.
        push!(E, expected(H, psi))
        push!(var, m_variance(H, psi))
        if debug > 0
            println("Done sweep $it, bond dimension: $current_D")
            @printf("    E: %.2e, ΔE: %.2e\n", E[it], E[it]-E[it-1])
            @printf("    var: %.2e, Δvar: %.2e\n", var[it], var[it]-var[it-1])
            @printf("    norm(psi): %.15e\n", contract(psi, psi))
        end

        # If variance converges enlarge bond dimension until it reaches `max_D`.
        if abs(var[it] - var[it-1])/var[it] < 1e-3 && current_D != D
            current_D = min(current_D + ceil(Int, D/5), D)
            if algorithm == "DMRG1"
                # If algorithm is 1-site we have to manually grow `D`.
                enlarge_bond_dimension!(psi, current_D)
            end
        end

        # Update while loop control parameters.
        var_is_stuck = abs(var[it] - var[it-1]) < 1e-8
        it += 1
    end

    return E, var
end

#
# 1-SITE DMRG.
#

"""
    update_lr_envs_1s!(i::Int, Ai::Vector{T}, psi::Mps{T}, H::Mpo{T},
                       Le::Vector{Array{T, 3}}, Re::Vector{Array{T, 3}},
                       sense::Int) where T<:Number

Update the left and right environments after the local Hamiltonian is minimized.
"""
function update_lr_envs_1s!(i::Int, Ai::Vector{T}, psi::Mps{T}, H::Mpo{T},
                            Le::Vector{Array{T, 3}}, Re::Vector{Array{T, 3}},
                            sense::Int) where T<:Number
    if sense == +1
        Ai = reshape(Ai, (size(Le[i], 1)*psi.d, size(Re[i], 1)))
        Qa, Ra = qr(Ai)
        Qa = Matrix(Qa)
        Qa = reshape(Qa, (size(Le[i], 1), psi.d, size(Qa, 2)))

        # Update left and right environments at L[i+1] and R[i] and psi.
        psi.A[i] = Qa
        if i < psi.L
            Le[i+1] = prop_right3(Le[i], Qa, H.M[i], Qa)
            @tensor Re[i][r1, r2, r3] = Ra[r1, a]*Ra[r3, b]*Re[i][a, r2, b]
        end
    else
        Ai = reshape(Ai, (size(Le[i], 1), psi.d*size(Re[i], 1)))
        La, Qa = lq(Ai)
        Qa = Matrix(Qa)
        Qa = reshape(Qa, (size(Qa, 1), psi.d, size(Re[i], 1)))

        # Update left and right environments at L[i] and R[i-1] and psi.
        psi.A[i] = Qa
        if i > 1
            Re[i-1] = prop_left3(Qa, H.M[i], Qa, Re[i])
            @tensor Le[i][l1, l2, l3] = Le[i][a, l2, b]*La[a, l1]*La[b, l3]
        end
    end
    return
end

"""
    do_sweep_1s!(psi::Mps{T}, H::Mpo{T},
                 Le::Vector{Array{T, 3}}, Re::Vector{Array{T, 3}},
                 sense::Int, debug::Int=0) where T<:Number

Do a sweep to locally minimize the energy of `psi` at 1 site per step. The
direction of the sweep is given by `sense = +1, -1`.
"""
function do_sweep_1s!(psi::Mps{T}, H::Mpo{T},
                      Le::Vector{Array{T, 3}}, Re::Vector{Array{T, 3}},
                      sense::Int, debug::Int=0) where T<:Number

    sense == 1 || sense == -1 || throw("`Sense` must be either `+1` or `-1`.")
    # Order of sites to do the sweep.
    sweep_sites = sense == +1 ? (1:psi.L) : reverse(1:psi.L)
    for i in sweep_sites
        # Compute local Hamiltonian.
        @tensor Hi[l1, s1, r1, l3, s2, r3] := (Le[i][l1, l2, l3]
                                               *H.M[i][l2, s1, s2, r2]
                                               *Re[i][r1, r2, r3])
        Hi = reshape(Hi, (size(Le[i], 1)*psi.d*size(Re[i], 1),
                          size(Le[i], 3)*psi.d*size(Re[i], 3)))
        # Compute lowest energy eigenvector of Hi.
        Ei, Mi = eigs(Hermitian(Hi), nev=1, which=:SR)
        Mi = vec(Mi)

        # Update left and right environments.
        update_lr_envs_1s!(i, Mi, psi, H, Le, Re, sense)

        # Useful debug information.
        debug > 1 && println("site: $i, size(Hi): $(size(Hi))")
    end
    return
end

#
# 2-SITE DMRG.
#

"""
    update_lr_envs_2s!(i::Int, Mi::Vector{T}, psi::Mps{T}, H::Mpo{T},
                       Le::Vector{Array{T, 3}}, Re::Vector{Array{T, 3}},
                       max_D::Int, sense::Int) where T<:Number

Update the left and right environments after the local Hamiltonian is minimized
with 2-site algorithm.
"""
function update_lr_envs_2s!(i::Int, Mi::Vector{T}, psi::Mps{T}, H::Mpo{T},
                            Le::Vector{Array{T, 3}}, Re::Vector{Array{T, 3}},
                            max_D::Int, sense::Int) where T<:Number
    # Decompose the Mi tensor spanning sites i and i+1 with SVD.
    Mi = reshape(Mi, size(psi.A[i], 1)*psi.d, psi.d*size(psi.A[i+1], 3))
    F = svd(Mi)
    # Keep the bond dimension stable by removing the lowest singular values.
    trim = min(max_D, length(F.S))
    svals = F.S[1:trim]
    # Divide sval by norm to keep state normalized.
    S = Diagonal(svals./norm(svals))
    if sense == +1
        U = F.U[:, 1:trim]
        Vt = S*F.Vt[1:trim, :]
    else
        U = F.U[:, 1:trim]*S
        Vt = F.Vt[1:trim, :]
    end
    U = reshape(U, size(psi.A[i], 1), psi.d, length(svals))
    Vt = reshape(Vt, length(svals), psi.d, size(psi.A[i+1], 3))
    # Update environments and state.
    Le[i+1] = prop_right3(Le[i], U, H.M[i], U)
    Re[i] = prop_left3(Vt, H.M[i+1], Vt, Re[i+1])
    psi.A[i] = U
    psi.A[i+1] = Vt
    return svals
end

"""
    do_sweep_2s!(psi::Mps{T}, H::Mpo{T},
                 Le::Vector{Array{T, 3}}, Re::Vector{Array{T, 3}},
                 sense::Int, max_D::Int, debug::Int=0) where T<:Number

Do a sweep to locally minimize the energy of `psi` at 2 sites per step. The
direction of the sweep is given by `sense = +1, -1`.
"""
function do_sweep_2s!(psi::Mps{T}, H::Mpo{T},
                      Le::Vector{Array{T, 3}}, Re::Vector{Array{T, 3}},
                      sense::Int, max_D::Int, debug::Int=0) where T<:Number

    sense == 1 || sense == -1 || throw("`Sense` must be either `+1` or `-1`.")
    # Order of sites to do the sweep.
    sweep_sites = sense == +1 ? (1:psi.L-1) : reverse(2:psi.L-1)
    for i in sweep_sites
        # Compute local Hamiltonian.
        @tensor Hi[l1, s1, s3, r1, l3, s2, s4, r3] := (Le[i][l1, l2, l3]
                                                       *H.M[i][l2, s1, s2, a]
                                                       *H.M[i+1][a, s3, s4, r2]
                                                       *Re[i+1][r1, r2, r3])
        Hi = reshape(Hi, (size(psi.A[i], 1)*psi.d^2*size(psi.A[i+1], 3),
                          size(psi.A[i], 1)*psi.d^2*size(psi.A[i+1], 3)))

        # Compute lowest energy eigenvector of Hi.
        Ei, Mi = eigs(Hermitian(Hi), nev=1, which=:SR)
        Mi = vec(Mi)

        # Update left and right environments.
        svals = update_lr_envs_2s!(i, Mi, psi, H, Le, Re, max_D, sense)

        # Useful debug information.
        if debug > 1
            println("Site: $i, size(Hi): $(size(Hi))")
            @printf("Sum discarded squared singular values: %.2e\n\n",
                    1. - norm(svals))
        end
    end
    return
end
