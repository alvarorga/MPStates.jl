export dmrg!

"""
    dmrg!(psi::Mps{T}, H::Mpo{T}, dmrg_opts::DMRGOpts) where {T<:Number}

Minimize energy of psi with respect to H using a DMRG algorithm.

Warning: the Hamiltonian is assumed to be Hermitian. This function will silently
fail if that is not the case.

Output:
    - `E::Vector{Float64}`: energy of psi after every sweep.
    - `var::Vector{Float64}`: variance of psi after every sweep.
"""
function dmrg!(psi::Mps{T}, H::Mpo{T}, dmrg_opts::DMRGOpts) where {T<:Number}
    # Create left and right environments.
    Le = fill(ones(T, 1, 1, 1), psi.L)
    Re = fill(ones(T, 1, 1, 1), psi.L)
    # Initialize left environment.
    for i=2:psi.L
        Le[i] = prop_right3(Le[i-1], psi.M[i-1], H.W[i-1], psi.M[i-1])
    end

    # Compute energy of state and variance at each sweep. Sweep 1 is just
    # the computation of the energy and the variance.
    E = Float64[]
    var = Float64[]
    # Noise term for DMRG3S.
    α = dmrg_opts.α
    # Cache to store temporaries.
    cache = Cache(Vector{AbstractArray{T}}())
    for s=1:dmrg_opts.nsweeps
        # Do left and right sweeps.
        time = @elapsed begin
        if dmrg_opts.algorithm == "DMRG1"
            Es = do_sweep_1s!(psi, H, Le, Re, -1, s, dmrg_opts)
            Es = do_sweep_1s!(psi, H, Le, Re, +1, s, dmrg_opts)
        elseif dmrg_opts.algorithm == "DMRG2"
            Es = do_sweep_2s!(psi, H, Le, Re, -1, s, cache, dmrg_opts)
            Es = do_sweep_2s!(psi, H, Le, Re, +1, s, cache, dmrg_opts)
        elseif dmrg_opts.algorithm == "DMRG3S"
            Es = do_sweep_3s!(psi, H, Le, Re, -1, s, cache, α, dmrg_opts)
            Es = do_sweep_3s!(psi, H, Le, Re, +1, s, cache, α, dmrg_opts)
        end
        end

        # Update energy and variance of psi after the sweep.
        push!(E, Es)
        push!(var, real(m_variance(H, psi)))

        # Print debug information: energy, variance and their variation.
        if dmrg_opts.show_trace > 0
            println("Done sweep $s, max bond dimension: $(max_bond_dim(psi))")
            @printf("    E: %.6e, ΔE: %.2e\n",
                    E[s], E[s]-(s > 1 ? E[s-1] : 1e5))
            @printf("    var: %.6e, Δvar: %.2e\n",
                    var[s], var[s]-(s > 1 ? var[s-1] : 1e5))
            println("    Elapsed time: $time s")
        end

        # Check for convergence of variance.
        if s > 1 && abs(var[s] - var[s-1]) < dmrg_opts.dmrg_tol
            println("The variance of psi has converged. Finished DMRG.")
            break
        end
    end
    return E, var
end

#
# 1-SITE DMRG.
#

"""
    update_lr_envs_1s!(psi::Mps{T},
                       H::Mpo{T},
                       Le::Vector{Array{T, 3}},
                       Re::Vector{Array{T, 3}},
                       sense::Int,
                       s::Int,
                       i::Int,
                       Mi::Vector{T},
                       dmrg_opts::DMRGOpts) where {T<:Number}

Update the left and right environments after the local Hamiltonian is minimized.
"""
function update_lr_envs_1s!(psi::Mps{T},
                            H::Mpo{T},
                            Le::Vector{Array{T, 3}},
                            Re::Vector{Array{T, 3}},
                            sense::Int,
                            s::Int,
                            i::Int,
                            Mi::Vector{T},
                            dmrg_opts::DMRGOpts) where {T<:Number}
    if sense == +1
        Mi = reshape(Mi, (size(Le[i], 1), psi.d, size(Re[i], 1)))
        psi.M[i], R = factorize_svd_right(Mi, cutoff=dmrg_opts.cutoff[s])

        # Update left and right environments at Le[i+1] and Re[i] and psi.
        psi.M[i] = psi.M[i]
        if i < psi.L
            Le[i+1] = prop_right3(Le[i], psi.M[i], H.W[i], psi.M[i])
            Re[i] = absorb_Re(Re[i], R)
        end
    else
        Mi = reshape(Mi, (size(Le[i], 1), psi.d, size(Re[i], 1)))
        L, psi.M[i] = factorize_svd_left(Mi, cutoff=dmrg_opts.cutoff[s])

        # Update left and right environments at Le[i] and Re[i-1] and psi.
        if i > 1
            Re[i-1] = prop_left3(psi.M[i], H.W[i], psi.M[i], Re[i])
            Le[i] = absorb_Le(Le[i], L)
        end
    end
    return psi
end

"""
    do_sweep_1s!(psi::Mps{T},
                 H::Mpo{T},
                 Le::Vector{Array{T, 3}},
                 Re::Vector{Array{T, 3}},
                 sense::Int,
                 s::Int,
                 dmrg_opts::DMRGOpts) where {T<:Number}

Do a sweep to locally minimize the energy of `psi` at 1 site per step. The
direction of the sweep is given by `sense = +1, -1`.
"""
function do_sweep_1s!(psi::Mps{T},
                      H::Mpo{T},
                      Le::Vector{Array{T, 3}},
                      Re::Vector{Array{T, 3}},
                      sense::Int,
                      s::Int,
                      dmrg_opts::DMRGOpts) where {T<:Number}

    # Manually increase the bond dimension.
    enlarge_bond_dimension!(psi, dmrg_opts.maxm[s])
    # Update left environment (right environment is updated in the first sweep).
    for i=2:psi.L
        Le[i] = prop_right3(Le[i-1], psi.M[i-1], H.W[i-1], psi.M[i-1])
    end

    sense == 1 || sense == -1 || throw("`Sense` must be either `+1` or `-1`.")
    # Energy before the sweep starts. Depends on the sweep sense, assume the
    # last sweep had the opposite sense.
    E = sense == +1 ?
        sum(real(Re[1].*Le[2])) : sum(real(Re[psi.L-1].*Le[psi.L]))

    # Order of sites to do the sweep.
    sweep_sites = sense == +1 ? (1:psi.L) : reverse(1:psi.L)
    for i in sweep_sites
        # Compute local minimum.
        Hi = build_local_hamiltonian(Le[i], H.W[i], Re[i])
        array_E, Mi = eigs(Hermitian(Hi), nev=1, which=:SR,
                           maxiter=dmrg_opts.lanczos_iters)
        E = real(array_E[1])
        Mi = vec(Mi)

        # Update left and right environments.
        update_lr_envs_1s!(psi, H, Le, Re, sense, s, i, Mi, dmrg_opts)

        # Useful debug information.
        if dmrg_opts.show_trace > 1
            println("site: $i, size(Hi): $(size(Hi)), Ei: $(E)")
        end
    end
    return E
end

#
# 2-SITE DMRG.
#

"""
    update_lr_envs_2s!(psi::Mps{T},
                       H::Mpo{T},
                       Le::Vector{Array{T, 3}},
                       Re::Vector{Array{T, 3}},
                       sense::Int,
                       s::Int,
                       i::Int,
                       Mi::Vector{T},
                       dmrg_opts::DMRGOpts) where {T<:Number}

Update the left and right environments after the local Hamiltonian is minimized
with 2-site algorithm.
"""
function update_lr_envs_2s!(psi::Mps{T},
                            H::Mpo{T},
                            Le::Vector{Array{T, 3}},
                            Re::Vector{Array{T, 3}},
                            sense::Int,
                            s::Int,
                            i::Int,
                            Mi::Vector{T},
                            dmrg_opts::DMRGOpts) where {T<:Number}

    # Decompose the Mi tensor spanning sites i and i+1 with SVD.
    Mi = reshape(Mi, size(psi.M[i], 1), psi.d, psi.d, size(psi.M[i+1], 3))
    dimcutoff = bond_dimension_with_m(psi.L, i+1, dmrg_opts.maxm[s], psi.d)
    psi.M[i], psi.M[i+1] = sense == 1 ?
        factorize_svd_right(Mi, dimcutoff=dimcutoff) :
        factorize_svd_left(Mi, dimcutoff=dimcutoff)

    # Update environments.
    Le[i+1] = prop_right3(Le[i], psi.M[i], H.W[i], psi.M[i])
    Re[i] = prop_left3(psi.M[i+1], H.W[i+1], psi.M[i+1], Re[i+1])
    return
end

"""
    do_sweep_2s!(psi::Mps{T},
                 H::Mpo{T},
                 Le::Vector{Array{T, 3}},
                 Re::Vector{Array{T, 3}},
                 sense::Int,
                 s::Int,
                 cache::Cache{T},
                 dmrg_opts::DMRGOpts) where {T<:Number}

Do a sweep to locally minimize the energy of `psi` at 2 sites per step. The
direction of the sweep is given by `sense = +1, -1`.
"""
function do_sweep_2s!(psi::Mps{T},
                      H::Mpo{T},
                      Le::Vector{Array{T, 3}},
                      Re::Vector{Array{T, 3}},
                      sense::Int,
                      s::Int,
                      cache::Cache{T},
                      dmrg_opts::DMRGOpts) where {T<:Number}

    sense == 1 || sense == -1 || throw("`Sense` must be either `+1` or `-1`.")
    # Energy before the sweep starts. Depends on the sweep sense, assume the
    # last sweep had the opposite sense.
    E = sense == +1 ?
        sum(real(Re[1].*Le[2])) : sum(real(Re[psi.L-1].*Le[psi.L]))

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
        array_E, Mi = eigs(Hermitian(Hi), nev=1, which=:SR, v0=v0,
                           maxiter=dmrg_opts.lanczos_iters)
        E = real(array_E[1])
        Mi = vec(Mi)

        # Update left and right environments.
        update_lr_envs_2s!(psi, H, Le, Re, sense, s, i, Mi, dmrg_opts)

        # Useful debug information.
        if dmrg_opts.show_trace > 1
            println("Site: $i, size(Hi): $(size(Hi)), Ei: $(E)")
        end
    end
    return E
end

#
# STRICTLY-SINGLE-SITE DMRG3S.
#

"""
    update_lr_envs_3s!(psi::Mps{T},
                       H::Mpo{T},
                       Le::Vector{Array{T, 3}},
                       Re::Vector{Array{T, 3}},
                       sense::Int,
                       s::Int,
                       i::Int,
                       Mi::Vector{T},
                       dmrg_opts::DMRGOpts) where {T<:Number}

Update the left and right environments after the local Hamiltonian is minimized.
"""
function update_lr_envs_3s!(psi::Mps{T},
                            H::Mpo{T},
                            Le::Vector{Array{T, 3}},
                            Re::Vector{Array{T, 3}},
                            sense::Int,
                            s::Int,
                            i::Int,
                            Mi::Vector{T},
                            α::Float64,
                            dmrg_opts::DMRGOpts) where {T<:Number}
    if sense == +1
        # Subspace expansion.
        Mi = reshape(Mi, (size(Le[i], 1), psi.d, size(Re[i], 1)))
        P = prop_right_subexp(Le[i], H.W[i], Mi)
        P = reshape(P, (size(Le[i], 1)*psi.d, size(Re[i], 2)*size(Re[i], 3)))
        # Add subpsace expansion to local mimimum.
        Mi = reshape(Mi, (size(Le[i], 1)*psi.d, size(Re[i], 1)))
        MP = hcat(Mi, α*P)

        # Svd.
        cutoff = dmrg_opts.cutoff[s]
        dimcutoff = bond_dimension_with_m(psi.L, i+1, dmrg_opts.maxm[s], psi.d)
        psi.M[i], SV = factorize_svd_right(reshape(MP, 1, size(MP)...),
                                           cutoff=cutoff, dimcutoff=dimcutoff)
        # Trim the dimension of SV to match with psi.M[i+1].
        SV = SV[:, 1:size(Re[i], 1)]
        # Unbind the bond dimension in psi.M[i].
        psi.M[i] = reshape(psi.M[i], size(Le[i], 1), psi.d, size(psi.M[i], 3))

        # Update left and right environments at Le[i+1] and Re[i] and psi.
        if i < psi.L
            Le[i+1] = prop_right3(Le[i], psi.M[i], H.W[i], psi.M[i])
            Re[i] = absorb_Re(Re[i], SV)
            # Update this tensor to provide a good initial vector later in eigs.
            psi.M[i+1] = absorb_fromleft(SV, psi.M[i+1])
        end
    else
        # Subspace expansion.
        Mi = reshape(Mi, (size(Le[i], 1), psi.d, size(Re[i], 1)))
        P = prop_left_subexp(H.W[i], Mi, Re[i])
        P = reshape(P, (psi.d*size(Re[i], 1), size(Le[i], 2)*size(Le[i], 3)))
        # Add subpsace expansion to local mimimum.
        Mi = reshape(Mi, (size(Le[i], 1), psi.d*size(Re[i], 1)))
        MP = vcat(Mi, α*transpose(P))

        # Svd.
        cutoff = dmrg_opts.cutoff[s]
        dimcutoff = bond_dimension_with_m(psi.L, i, dmrg_opts.maxm[s], psi.d)
        US, psi.M[i] = factorize_svd_left(reshape(MP, size(MP)..., 1),
                                          cutoff=cutoff, dimcutoff=dimcutoff)
        # Trim the dimension of US to match with psi.M[i-1].
        US = US[1:size(Le[i], 1), :]
        # Unbind the bond dimension in psi.M[i].
        psi.M[i] = reshape(psi.M[i], size(psi.M[i], 1), psi.d, size(Re[i], 1))

        # Update left and right environments at Le[i+1] and Re[i] and psi.
        if i > 1
            Re[i-1] = prop_left3(psi.M[i], H.W[i], psi.M[i], Re[i])
            Le[i] = absorb_Le(Le[i], US)
            # Update this tensor to provide a good initial vector later in eigs.
            psi.M[i-1] = absorb_fromright(psi.M[i-1], US)
        end
    end
    return psi
end

"""
    do_sweep_3s!(psi::Mps{T},
                 H::Mpo{T},
                 Le::Vector{Array{T, 3}},
                 Re::Vector{Array{T, 3}},
                 sense::Int,
                 s::Int,
                 cache::Cache{T},
                 dmrg_opts::DMRGOpts) where {T<:Number}

Do a sweep to locally minimize the energy of `psi` at 1 site per step. The
direction of the sweep is given by `sense = +1, -1`.
"""
function do_sweep_3s!(psi::Mps{T},
                      H::Mpo{T},
                      Le::Vector{Array{T, 3}},
                      Re::Vector{Array{T, 3}},
                      sense::Int,
                      s::Int,
                      cache::Cache{T},
                      α::Float64,
                      dmrg_opts::DMRGOpts) where {T<:Number}

    sense == 1 || sense == -1 || throw("`Sense` must be either `+1` or `-1`.")
    # Energy before the sweep starts. Depends on the sweep sense, assume the
    # last sweep had the opposite sense.
    E = sense == +1 ?
        sum(real(Re[1].*Le[2])) : sum(real(Re[psi.L-1].*Le[psi.L]))

    # Order of sites to do the sweep.
    sweep_sites = sense == +1 ? (1:psi.L-1) : reverse(2:psi.L)
    for i in sweep_sites
        # Compute local minimum.
        Hi = build_local_hamiltonian(Le[i], H.W[i], Re[i], cache)
        v0 = vec(deepcopy(psi.M[i]))
        array_E, Mi = eigs(Hermitian(Hi), nev=1, which=:SR, v0=v0,
                           maxiter=dmrg_opts.lanczos_iters)
        E1 = real(array_E[1])
        delta_E1 = E1-E
        Mi = vec(Mi)

        # Update left and right environments.
        update_lr_envs_3s!(psi, H, Le, Re, sense, s, i, Mi, α, dmrg_opts)

        # Compute new energy and update α.
        E = sense == +1 ? sum(real(Re[i].*Le[i+1])) : sum(real(Re[i-1].*Le[i]))
        delta_E = E-E1
        if -delta_E/delta_E1 > 0.3/log(dmrg_opts.maxm[s])
            α /= 2.
        elseif 0. < -delta_E/delta_E1 < min(1/dmrg_opts.maxm[s], 0.1)
            α *= 2.
        end

        # Useful debug information.
        if dmrg_opts.show_trace > 1
            println("site: $i, size(Hi): $(size(Hi)), Ei: $(E), α: $(dmrg_opts.α)")
        end
        if dmrg_opts.show_trace > 2
            @printf("ΔE_0: %.3e, ΔE_0/ΔE_T: %.3e\n", delta_E1, delta_E/delta_E1)
        end
    end
    return E
end
