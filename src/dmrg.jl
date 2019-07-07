#
# DMRG algorithms.
#

"""
    minimize!(psi::Mps{T}, H::Mpo{T}, D::Int, algorithm::String="DMRG1";
              max_iters::Int=500, tol::Float64=1e-6,
              debug::Int=1) where T<:Number

Minimize energy of `psi` with respect to `H`, allowing maximum bond dimension
`D`.

Return the energy and variance of `psi` at every sweep. The `debug` parameter
controls the amount of information the script outputs at runtime. Possible
algorithms to minimize the energy are: "DMRG1": 1-site DMRG.
"""
function minimize!(psi::Mps{T}, H::Mpo{T}, D::Int, algorithm::String="DMRG1";
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
        Le[i] = prop_right3(Le[i-1], psi.M[i-1], H.W[i-1], psi.M[i-1])
    end

    # Compute energy of state and variance at each sweep. We start at iteration
    # `it` = 2, starting with artificially big values in the first iteration.
    alpha = 1e-6
    E = [1e5]
    var = [1e5]
    it = 2
    var_is_stuck = false
    while var[it-1] > tol && it <= max_iters+1 && !var_is_stuck
        # Do left and right sweeps.
        if algorithm == "DMRG1"
            do_sweep_1s!(psi, H, Le, Re, -1, debug)
            E_sweep = do_sweep_1s!(psi, H, Le, Re, +1, debug)
        elseif algorithm == "DMRG2"
            do_sweep_2s!(psi, H, Le, Re, -1, current_D, debug)
            E_sweep = do_sweep_2s!(psi, H, Le, Re, +1, current_D, debug)
        elseif algorithm == "DMRG3S"
            E_sweep, alpha = do_sweep_3s!(psi, H, Le, Re, -1, current_D, alpha, debug)
            E_sweep, alpha = do_sweep_3s!(psi, H, Le, Re, +1, current_D, alpha, debug)
        end

        # Compute energy and variance of `psi` after sweeps.
        push!(E, E_sweep)
        push!(var, real(m_variance(H, psi)))
        if debug > 0
            println("Done sweep $it, bond dimension: $current_D")
            @printf("    E: %.6e, ΔE: %.2e\n", E[it], E[it]-E[it-1])
            @printf("    var: %.6e, Δvar: %.2e\n", var[it], var[it]-var[it-1])
            @printf("    norm(psi): %.15e\n", contract(psi, psi))
        end

        # If variance converges enlarge bond dimension until it reaches `max_D`.
        if abs(var[it] - var[it-1])/var[it] < 1e-2 && current_D != D
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
    update_lr_envs_1s!(psi::Mps{T}, i::Int, Mi::Vector{T}, H::Mpo{T},
                       Le::Vector{Array{T, 3}}, Re::Vector{Array{T, 3}},
                       sense::Int) where T<:Number

Update the left and right environments after the local Hamiltonian is minimized.
"""
function update_lr_envs_1s!(psi::Mps{T}, i::Int, Mi::Vector{T}, H::Mpo{T},
                            Le::Vector{Array{T, 3}}, Re::Vector{Array{T, 3}},
                            sense::Int) where T<:Number
    if sense == +1
        Mi = reshape(Mi, (size(Le[i], 1)*psi.d, size(Re[i], 1)))
        Ai, R = qr(Mi)
        Ai = Matrix(Ai)
        Ai = reshape(Ai, (size(Le[i], 1), psi.d, size(Ai, 2)))

        # Update left and right environments at Le[i+1] and Re[i] and psi.
        psi.M[i] = Ai
        if i < psi.L
            Le[i+1] = prop_right3(Le[i], Ai, H.W[i], Ai)
            @tensor Re[i][r1, r2, r3] = R[r1, a]*conj(R[r3, b])*Re[i][a, r2, b]
        end
    else
        Mi = reshape(Mi, (size(Le[i], 1), psi.d*size(Re[i], 1)))
        L, Bi = lq(Mi)
        Bi = Matrix(Bi)
        Bi = reshape(Bi, (size(Bi, 1), psi.d, size(Re[i], 1)))

        # Update left and right environments at Le[i] and Re[i-1] and psi.
        psi.M[i] = Bi
        if i > 1
            Re[i-1] = prop_left3(Bi, H.W[i], Bi, Re[i])
            @tensor Le[i][l1, l2, l3] = Le[i][a, l2, b]*L[a, l1]*conj(L[b, l3])
        end
    end
    return psi
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

    # Energy after the sweep.
    E = 0.
    sense == 1 || sense == -1 || throw("`Sense` must be either `+1` or `-1`.")
    # Order of sites to do the sweep.
    sweep_sites = sense == +1 ? (1:psi.L) : reverse(1:psi.L)
    for i in sweep_sites
        # Compute local Hamiltonian.
        @tensor Hi[l1, s1, r1, l3, s2, r3] := (Le[i][l1, l2, l3]
                                               *H.W[i][l2, s1, s2, r2]
                                               *Re[i][r1, r2, r3])
        Hi = reshape(Hi, (size(Le[i], 1)*psi.d*size(Re[i], 1),
                          size(Le[i], 3)*psi.d*size(Re[i], 3)))
        # Compute lowest energy eigenvector of Hi.
        array_E, Mi = eigs(Hermitian(Hi), nev=1, which=:SR)
        E = real(array_E[1])
        Mi = vec(Mi)

        # Update left and right environments.
        update_lr_envs_1s!(psi, i, Mi, H, Le, Re, sense)

        # Useful debug information.
        debug > 1 && println("site: $i, size(Hi): $(size(Hi)), Ei: $(E)")
    end
    return E
end

#
# 2-SITE DMRG.
#

"""
    update_lr_envs_2s!(psi::Mps{T}, i::Int, Mi::Vector{T}, H::Mpo{T},
                       Le::Vector{Array{T, 3}}, Re::Vector{Array{T, 3}},
                       max_D::Int, sense::Int) where T<:Number

Update the left and right environments after the local Hamiltonian is minimized
with 2-site algorithm.
"""
function update_lr_envs_2s!(psi::Mps{T}, i::Int, Mi::Vector{T}, H::Mpo{T},
                            Le::Vector{Array{T, 3}}, Re::Vector{Array{T, 3}},
                            max_D::Int, sense::Int) where T<:Number
    # Decompose the Mi tensor spanning sites i and i+1 with SVD.
    Mi = reshape(Mi, size(psi.M[i], 1)*psi.d, psi.d*size(psi.M[i+1], 3))
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
    U = reshape(U, size(psi.M[i], 1), psi.d, length(svals))
    Vt = reshape(Vt, length(svals), psi.d, size(psi.M[i+1], 3))
    # Update environments and state.
    Le[i+1] = prop_right3(Le[i], U, H.W[i], U)
    Re[i] = prop_left3(Vt, H.W[i+1], Vt, Re[i+1])
    psi.M[i] = U
    psi.M[i+1] = Vt
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

    # Energy after sweep.
    E = 0.
    sense == 1 || sense == -1 || throw("`Sense` must be either `+1` or `-1`.")
    # Order of sites to do the sweep.
    sweep_sites = sense == +1 ? (1:psi.L-1) : reverse(2:psi.L-1)
    for i in sweep_sites
        # Compute local Hamiltonian.
        @tensor Hi[l1, s1, s3, r1, l3, s2, s4, r3] := (Le[i][l1, l2, l3]
                                                       *H.W[i][l2, s1, s2, a]
                                                       *H.W[i+1][a, s3, s4, r2]
                                                       *Re[i+1][r1, r2, r3])
        Hi = reshape(Hi, (size(psi.M[i], 1)*psi.d^2*size(psi.M[i+1], 3),
                          size(psi.M[i], 1)*psi.d^2*size(psi.M[i+1], 3)))

        # Compute lowest energy eigenvector of Hi.
        array_E, Mi = eigs(Hermitian(Hi), nev=1, which=:SR)
        E = real(array_E[1])
        Mi = vec(Mi)

        # Update left and right environments.
        svals = update_lr_envs_2s!(psi, i, Mi, H, Le, Re, max_D, sense)

        # Useful debug information.
        if debug > 1
            println("Site: $i, size(Hi): $(size(Hi)), Ei: $(E)")
            @printf("Sum of discarded squared singular values: %.2e\n\n",
                    1. - norm(svals))
        end
    end
    return E
end

#
# STRICTLY-SINGLE-SITE DMRG3S.
#

"""
    update_lr_envs_3s!(psi::Mps{T}, i::Int, Mi::Vector{T}, H::Mpo{T},
                       Le::Vector{Array{T, 3}}, Re::Vector{Array{T, 3}},
                       m::Int, alpha::Float64, sense::Int) where T<:Number

Update the left and right environments after the local Hamiltonian is minimized.
"""
function update_lr_envs_3s!(psi::Mps{T}, i::Int, Mi::Vector{T}, H::Mpo{T},
                            Le::Vector{Array{T, 3}}, Re::Vector{Array{T, 3}},
                            m::Int, alpha::Float64, sense::Int) where T<:Number
    if sense == +1
        # Subspace expansion.
        Mi = reshape(Mi, (size(Le[i], 1), psi.d, size(Re[i], 1)))
        P = prop_right_subexp(Le[i], H.W[i], Mi)
        P = reshape(P, (size(Le[i], 1)*psi.d, size(Re[i], 2)*size(Re[i], 3)))
        # Add subpsace expansion to local mimimum.
        Mi = reshape(Mi, (size(Le[i], 1)*psi.d, size(Re[i], 1)))
        MP = hcat(Mi, alpha*P)

        # Svd.
        F = svd(MP)
        # Trim SVD to the desired bond dimension.
        new_m = min(bond_dimension_with_m(psi.L, i+1, m, psi.d), length(F.S))
        U = F.U[:, 1:new_m]
        svals = F.S[1:new_m]
        # Divide sval by norm to keep state normalized.
        S = Diagonal(svals./norm(svals))
        # Trim SV in the right part to match the size of psi.M[i+1].
        SV = S*F.Vt[1:new_m, 1:size(Re[i], 1)]

        # Update left and right environments at Le[i+1] and Re[i] and psi.
        Ai = reshape(U, (size(Le[i], 1), psi.d, new_m))
        psi.M[i] = Ai
        if i < psi.L
            Le[i+1] = prop_right3(Le[i], Ai, H.W[i], Ai)
            # Absorb SV into psi.M[i+1].
            Ci = reshape(psi.M[i+1], (size(psi.M[i+1], 1), psi.d*size(psi.M[i+1], 3)))
            Ci = SV*Ci
            psi.M[i+1] = reshape(Ci, (new_m, psi.d, size(Re[i+1], 1)))
            @tensor Re[i][r1, r2, r3] = SV[r1, a]*conj(SV[r3, b])*Re[i][a, r2, b]
        end
    else
        # Subspace expansion.
        Mi = reshape(Mi, (size(Le[i], 1), psi.d, size(Re[i], 1)))
        P = prop_left_subexp(H.W[i], Mi, Re[i])
        P = reshape(P, (psi.d*size(Re[i], 1), size(Le[i], 2)*size(Le[i], 3)))
        # Add subpsace expansion to local mimimum.
        Mi = reshape(Mi, (size(Le[i], 1), psi.d*size(Re[i], 1)))
        MP = vcat(Mi, alpha*transpose(P))

        # Svd.
        F = svd(MP)
        # Trim SVD to the desired bond dimension.
        new_m = min(bond_dimension_with_m(psi.L, i, m, psi.d), length(F.S))
        Vt = F.Vt[1:new_m, :]
        svals = F.S[1:new_m]
        # Divide sval by norm to keep state normalized.
        S = Diagonal(svals./norm(svals))
        # Trim US in the left part to match the size of psi.M[i-1].
        US = F.U[1:size(Le[i], 1), 1:new_m]*S

        # Update left and right environments at Le[i+1] and Re[i] and psi.
        Bi = reshape(Vt, (new_m, psi.d, size(Re[i], 1)))
        psi.M[i] = Bi
        if i > 1
            Re[i-1] = prop_left3(Bi, H.W[i], Bi, Re[i])
            # Absorb US into psi.M[i-1].
            Ci = reshape(psi.M[i-1], (size(psi.M[i-1], 1)*psi.d, size(psi.M[i-1], 3)))
            Ci = Ci*US
            psi.M[i-1] = reshape(Ci, (size(Le[i-1], 1), psi.d, new_m))
            @tensor Le[i][l1, l2, l3] := Le[i][a, l2, b]*US[a, l1]*conj(US[b, l3])
        end
    end
    return psi
end

"""
    do_sweep_3s!(psi::Mps{T}, H::Mpo{T},
                 Le::Vector{Array{T, 3}}, Re::Vector{Array{T, 3}},
                 sense::Int, m::Int, alpha::Float64,
                 debug::Int=0) where T<:Number

Do a sweep to locally minimize the energy of `psi` at 1 site per step. The
direction of the sweep is given by `sense = +1, -1`.
"""
function do_sweep_3s!(psi::Mps{T}, H::Mpo{T},
                      Le::Vector{Array{T, 3}}, Re::Vector{Array{T, 3}},
                      sense::Int, m::Int, alpha::Float64,
                      debug::Int=0) where T<:Number

    # Energy after the sweep.
    local E = 0.
    sense == 1 || sense == -1 || throw("`Sense` must be either `+1` or `-1`.")
    # Order of sites to do the sweep.
    sweep_sites = sense == +1 ? (1:psi.L-1) : reverse(2:psi.L)
    for i in sweep_sites
        # Compute local Hamiltonian.
        @tensor Hi[l1, s1, r1, l3, s2, r3] := (Le[i][l1, l2, l3]
                                               *H.W[i][l2, s1, s2, r2]
                                               *Re[i][r1, r2, r3])
        Hi = reshape(Hi, (size(Le[i], 1)*psi.d*size(Re[i], 1),
                          size(Le[i], 3)*psi.d*size(Re[i], 3)))
        # Compute lowest energy eigenvector of Hi.
        array_E, Mi = eigs(Hermitian(Hi), nev=1, which=:SR)
        E1 = real(array_E[1])
        delta_E1 = E1-E
        Mi = vec(Mi)

        # Update left and right environments.
        update_lr_envs_3s!(psi, i, Mi, H, Le, Re, m, alpha, sense)

        # Compute new energy and update alpha.
        if sense == +1
            E = sum(Re[i].*Le[i+1])
        else
            E = sum(Re[i-1].*Le[i])
        end
        delta_E = E-E1
        debug > 2 && @printf("ΔE_0: %.3e, ΔE_0/ΔE_T: %.3e\n", delta_E1, delta_E/delta_E1)
        if -delta_E/delta_E1 > 0.3
            alpha /= 2.
        elseif -delta_E/delta_E1 < 0.1
            alpha *= 2.
        end

        # Useful debug information.
        debug > 1 && println("site: $i, size(Hi): $(size(Hi)), Ei: $(E), alpha: $(alpha)")
    end
    return E, alpha
end
