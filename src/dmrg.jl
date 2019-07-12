#
# DMRG algorithms.
#

"""
    minimize!(psi::Mps{T}, H::Mpo{T}, max_m::Int, algorithm::String="DMRG1";
              max_iters::Int=500, tol::Float64=1e-6,
              debug::Int=1) where T<:Number

Minimize energy of `psi` with respect to `H`, allowing maximum bond dimension
`D`.

Return the energy and variance of `psi` at every sweep. The `debug` parameter
controls the amount of information the script outputs at runtime. Possible
algorithms to minimize the energy are: "DMRG1": 1-site DMRG.
"""
function minimize!(psi::Mps{T}, H::Mpo{T}, max_m::Int, algorithm::String="DMRG1";
                   max_iters::Int=500, tol::Float64=1e-6,
                   debug::Int=1) where T<:Number
    # Initialize the bond dimension at a sufficiently large number.
    m = max(8, maximum(size.(psi.M, 3)))
    # Manually increase bond dimension for DMRG1.
    if algorithm == "DMRG1"
        enlarge_bond_dimension!(psi, m)
    end

    # Create left and right environments.
    Le = fill(ones(T, 1, 1, 1), psi.L)
    Re = fill(ones(T, 1, 1, 1), psi.L)
    # Initialize left environment.
    for i=2:psi.L
        Le[i] = prop_right3(Le[i-1], psi.M[i-1], H.W[i-1], psi.M[i-1])
    end

    # Compute energy of state and variance at each sweep. Iteration 1 is just
    # the computation of the energy and the variance.
    alpha = 1e-6
    E = [expected(H, psi)]
    var = [m_variance(H, psi)]
    it = 2
    var_is_stuck = false
    cache = Cache(Vector{AbstractArray{T}}())
    while var[it-1] > tol && it <= max_iters && !var_is_stuck
        # Do left and right sweeps.
        if algorithm == "DMRG1"
            Es = do_sweep_1s!(psi, H, Le, Re, -1, debug)
            Es = do_sweep_1s!(psi, H, Le, Re, +1, debug)
        elseif algorithm == "DMRG2"
            Es = do_sweep_2s!(psi, H, Le, Re, -1, m, cache, debug)
            Es = do_sweep_2s!(psi, H, Le, Re, +1, m, cache, debug)
        elseif algorithm == "DMRG3S"
            Es, alpha = do_sweep_3s!(psi, H, Le, Re, -1, m, alpha, cache, debug)
            Es, alpha = do_sweep_3s!(psi, H, Le, Re, +1, m, alpha, cache, debug)
        end

        # Compute energy and variance of `psi` after sweeps.
        push!(E, Es)
        push!(var, real(m_variance(H, psi)))
        if debug > 0
            println("Done sweep $it, bond dimension: $m")
            @printf("    E: %.6e, ΔE: %.2e\n", E[it], E[it]-E[it-1])
            @printf("    var: %.6e, Δvar: %.2e\n", var[it], var[it]-var[it-1])
            @printf("    1-norm(psi): %.2e\n", 1. - norm(psi))
        end

        # If variance converges enlarge bond dimension until it reaches `max_m`.
        if abs(var[it] - var[it-1])/var[it] < 1e-2 && m < max_m
            m = min(m*psi.d, max_m)
            # For DMRG1 the bond dimension is grown manually.
            if algorithm == "DMRG1"
                enlarge_bond_dimension!(psi, m)
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
            Re[i] = absorb_Re(Re[i], R)
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
            Le[i] = absorb_Le(Le[i], L)
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
        # Compute local minimum.
        Hi = build_local_hamiltonian(Le[i], H.W[i], Re[i])
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
                       m::Int, sense::Int) where T<:Number

Update the left and right environments after the local Hamiltonian is minimized
with 2-site algorithm.
"""
function update_lr_envs_2s!(psi::Mps{T}, i::Int, Mi::Vector{T}, H::Mpo{T},
                            Le::Vector{Array{T, 3}}, Re::Vector{Array{T, 3}},
                            m::Int, sense::Int) where T<:Number
    # Decompose the Mi tensor spanning sites i and i+1 with SVD.
    Mi = reshape(Mi, size(psi.M[i], 1)*psi.d, psi.d*size(psi.M[i+1], 3))
    F = svd!(Mi)
    # Keep the bond dimension stable by removing the lowest singular values.
    trim = min(m, length(F.S))
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
                 sense::Int, m::Int,
                 cache::Cache{T}, debug::Int=0) where T<:Number

Do a sweep to locally minimize the energy of `psi` at 2 sites per step. The
direction of the sweep is given by `sense = +1, -1`.
"""
function do_sweep_2s!(psi::Mps{T}, H::Mpo{T},
                      Le::Vector{Array{T, 3}}, Re::Vector{Array{T, 3}},
                      sense::Int, m::Int,
                      cache::Cache{T}, debug::Int=0) where T<:Number

    # Energy after sweep.
    E = 0.
    sense == 1 || sense == -1 || throw("`Sense` must be either `+1` or `-1`.")
    # Order of sites to do the sweep.
    sweep_sites = sense == +1 ? (1:psi.L-1) : reverse(2:psi.L-1)
    for i in sweep_sites
        # Compute local minimum.
        Hi = build_local_hamiltonian_2(Le[i], H.W[i], H.W[i+1], Re[i+1], cache)
        # Build an initial vector for eigs using the previous M tensors.
        M1 = reshape(deepcopy(psi.M[i]),
                     (size(psi.M[i], 1)*size(psi.M[i], 2), size(psi.M[i], 3)))
        M2 = reshape(deepcopy(psi.M[i+1]),
                     (size(psi.M[i+1], 1), size(psi.M[i+1], 2)*size(psi.M[i+1], 3)))
        v0 = vec(M1*M2)
        array_E, Mi = eigs(Hermitian(Hi), nev=1, which=:SR, v0=v0)
        E = real(array_E[1])
        Mi = vec(Mi)

        # Update left and right environments.
        svals = update_lr_envs_2s!(psi, i, Mi, H, Le, Re, m, sense)

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
        F = svd!(MP)
        # Trim SVD to the desired bond dimension.
        new_m = min(bond_dimension_with_m(psi.L, i+1, m, psi.d), length(F.S))
        U = F.U[:, 1:new_m]
        svals = F.S[1:new_m]
        S = Diagonal(svals)
        # Trim SV in the right part to match the size of psi.M[i+1].
        SV = S*F.Vt[1:new_m, 1:size(Re[i], 1)]
        # Divide SV by norm to keep state normalized.
        SV ./= norm(SV)

        # Update left and right environments at Le[i+1] and Re[i] and psi.
        Ai = reshape(U, (size(Le[i], 1), psi.d, new_m))
        psi.M[i] = Ai
        if i < psi.L
            Le[i+1] = prop_right3(Le[i], Ai, H.W[i], Ai)
            # Absorb SV into psi.M[i+1].
            Ci = reshape(psi.M[i+1], (size(psi.M[i+1], 1), psi.d*size(psi.M[i+1], 3)))
            Ci = SV*Ci
            psi.M[i+1] = reshape(Ci, (new_m, psi.d, size(Re[i+1], 1)))
            Re[i] = absorb_Re(Re[i], SV)
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
        F = svd!(MP)
        # Trim SVD to the desired bond dimension.
        new_m = min(bond_dimension_with_m(psi.L, i, m, psi.d), length(F.S))
        Vt = F.Vt[1:new_m, :]
        svals = F.S[1:new_m]
        S = Diagonal(svals)
        # Trim US in the left part to match the size of psi.M[i-1].
        US = F.U[1:size(Le[i], 1), 1:new_m]*S
        # Divide US by norm to keep state normalized.
        US ./= norm(US)

        # Update left and right environments at Le[i+1] and Re[i] and psi.
        Bi = reshape(Vt, (new_m, psi.d, size(Re[i], 1)))
        psi.M[i] = Bi
        if i > 1
            Re[i-1] = prop_left3(Bi, H.W[i], Bi, Re[i])
            # Absorb US into psi.M[i-1].
            Ci = reshape(psi.M[i-1], (size(psi.M[i-1], 1)*psi.d, size(psi.M[i-1], 3)))
            Ci = Ci*US
            psi.M[i-1] = reshape(Ci, (size(Le[i-1], 1), psi.d, new_m))
            Le[i] = absorb_Le(Le[i], US)
        end
    end
    return psi
end

"""
    do_sweep_3s!(psi::Mps{T}, H::Mpo{T},
                 Le::Vector{Array{T, 3}}, Re::Vector{Array{T, 3}},
                 sense::Int, m::Int, alpha::Float64,
                 cache::Cache{T}, debug::Int=0) where T<:Number

Do a sweep to locally minimize the energy of `psi` at 1 site per step. The
direction of the sweep is given by `sense = +1, -1`.
"""
function do_sweep_3s!(psi::Mps{T}, H::Mpo{T},
                      Le::Vector{Array{T, 3}}, Re::Vector{Array{T, 3}},
                      sense::Int, m::Int, alpha::Float64,
                      cache::Cache{T}, debug::Int=0) where T<:Number
    # Energy after the sweep.
    E = 0.
    sense == 1 || sense == -1 || throw("`Sense` must be either `+1` or `-1`.")
    # Order of sites to do the sweep.
    sweep_sites = sense == +1 ? (1:psi.L-1) : reverse(2:psi.L)
    for i in sweep_sites
        # Compute local minimum.
        Hi = build_local_hamiltonian(Le[i], H.W[i], Re[i], cache)
        v0 = vec(deepcopy(psi.M[i]))
        array_E, Mi = eigs(Hermitian(Hi), nev=1, which=:SR, v0=v0)
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
