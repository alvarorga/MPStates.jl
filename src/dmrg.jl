export MinimizeOpts,
       minimize!

"""
Options for the DMRG minimization algorithm.

Fields:
    algorithm::String: DMRG algorithm used for minimizing the state. Can be:
        "DMRG1" for one site DMRG, "DMRG2" for two site DMRG, and "DMRG3S" for
        strictly single site DMRG with subspace expansion.
    tol::Float64: stop the algorithm when the change in the variance of the
        state is less than `tol`.
    max_sweeps::Int: maximum number of sweeps allowed.
    debug::Int: output information at every step of the minmization:
        - 0: no info given.
        - 1: energy, variance and their variation with respect to their last
            value after every right + left sweep.
        - 2: energy and size of the local Hamiltonian at every step of every
            sweep.
    sweep_dims::Vector{Int}: maximum bond dimension at every sweep.
"""
struct MinimizeOpts
    algorithm::String
    tol::Float64
    max_sweeps::Int
    debug::Int
    sweep_dims::Vector{Int}
end

"""
    MinimizeOpts(m::Int, algorithm::String; debug=0)

Initialize MinimizeOpts for a maximum bond dimension `m`.
"""
function MinimizeOpts(m::Int, algorithm::String; debug=0)
    tol = 1e-6
    max_sweeps = 200
    # We fill `sweep_dims` starting with m=8 and doubling the bond dimension
    # each five sweeps until the maximum bond dimension `m` is reached.
    sweep_dims = fill(m, max_sweeps)
    mi = 8
    cont = 1
    while mi < m
        sweep_dims[cont:cont+4] .= mi
        mi *= 2
        cont += 5
    end
    return MinimizeOpts(algorithm, tol, max_sweeps, debug, sweep_dims)
end

"""
    MinimizeOpts(sweep_dims::Vector{Int}, algorithm::String; debug=0)

Initialize MinimizeOpts with an specified set of bond dimensions for all
sweeps.
"""
function MinimizeOpts(sweep_dims::Vector{Int}, algorithm::String; debug=0)
    tol = 1e-6
    return MinimizeOpts(algorithm, tol, length(sweep_dims), debug, sweep_dims)
end

"""
    minimize!(psi::Mps{T}, H::Mpo{T}, minopts::MinimizeOpts) where T<:Number

Minimize energy of `psi` with respect to `H`.

Warning: the Hamiltonian is assumed to be Hermitian. This function will silently
fail if that is not the case.

Output:
    - `E`: energy of `psi` after every right + left sweep.
    - `var`: variance of `psi` after every right + left sweep.
"""
function minimize!(psi::Mps{T}, H::Mpo{T}, min_opts::MinimizeOpts) where T<:Number

    # Create left and right environments.
    Le = fill(ones(T, 1, 1, 1), psi.L)
    Re = fill(ones(T, 1, 1, 1), psi.L)
    # Initialize left environment.
    for i=2:psi.L
        Le[i] = prop_right3(Le[i-1], psi.M[i-1], H.W[i-1], psi.M[i-1])
    end

    # Compute energy of state and variance at each sweep. Iteration 1 is just
    # the computation of the energy and the variance.
    # Parameter to control subspace expansion in "DMRG3S" algorithm.
    α = 1e-6
    E = [real(expected(H, psi))]
    var = [m_variance(H, psi)]
    it = 2
    # Variable to observe if the variance has stopped decreasing at all.
    var_is_stuck = false
    # Variable to see if the tolerace of the variance has been reached.
    reached_tol = false
    # Cache to store temporaries.
    cache = Cache(Vector{AbstractArray{T}}())

    while !reached_tol && it <= min_opts.max_sweeps && !var_is_stuck
        # Maximum bond dimension in this sweep.
        m = min_opts.sweep_dims[it]
        # Do left and right sweeps.
        if min_opts.algorithm == "DMRG1"
            Es = do_sweep_1s!(psi, H, Le, Re, -1, m, min_opts.debug)
            Es = do_sweep_1s!(psi, H, Le, Re, +1, m, min_opts.debug)
        elseif min_opts.algorithm == "DMRG2"
            Es = do_sweep_2s!(psi, H, Le, Re, -1, m, cache, min_opts.debug)
            Es = do_sweep_2s!(psi, H, Le, Re, +1, m, cache, min_opts.debug)
        elseif min_opts.algorithm == "DMRG3S"
            Es, α = do_sweep_3s!(psi, H, Le, Re, -1, m, α, cache, min_opts.debug)
            Es, α = do_sweep_3s!(psi, H, Le, Re, +1, m, α, cache, min_opts.debug)
        end

        # Update energy and variance of `psi` after the sweep.
        push!(E, Es)
        push!(var, real(m_variance(H, psi)))

        # Print debug information: energy, variance and their variation.
        if min_opts.debug > 0
            println("Done sweep $it, bond dimension: $m")
            @printf("    E: %.6e, ΔE: %.2e\n", E[it], E[it]-E[it-1])
            @printf("    var: %.6e, Δvar: %.2e\n", var[it], var[it]-var[it-1])
        end

        # Update minimization loop control parameters. Prevent stopping if
        # maximum bond dimension has not been reached yet.
        var_is_stuck = abs(var[it] - var[it-1]) < 1e-8 && m == maximum(min_opts.sweep_dims)
        reached_tol = var[it] <= min_opts.tol && m == maximum(min_opts.sweep_dims)
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
        Mi = reshape(Mi, (size(Le[i], 1), psi.d, size(Re[i], 1)))
        Ai, R = factorize_svd_right(Mi, cutoff=0.)

        # Update left and right environments at Le[i+1] and Re[i] and psi.
        psi.M[i] = Ai
        if i < psi.L
            Le[i+1] = prop_right3(Le[i], Ai, H.W[i], Ai)
            Re[i] = absorb_Re(Re[i], R)
        end
    else
        Mi = reshape(Mi, (size(Le[i], 1), psi.d, size(Re[i], 1)))
        L, Bi = factorize_svd_left(Mi, cutoff=0.)

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
                 sense::Int, m::Int, debug::Int=0) where T<:Number

Do a sweep to locally minimize the energy of `psi` at 1 site per step. The
direction of the sweep is given by `sense = +1, -1`.
"""
function do_sweep_1s!(psi::Mps{T}, H::Mpo{T},
                      Le::Vector{Array{T, 3}}, Re::Vector{Array{T, 3}},
                      sense::Int, m::Int, debug::Int=0) where T<:Number

    # Manually increase the bond dimension.
    enlarge_bond_dimension!(psi, m)
    # Update right environment (left environment is updated in the first sweep).
    for i=reverse(1:psi.L-1)
        Re[i] = prop_left3(psi.M[i+1], H.W[i+1], psi.M[i+1], Re[i+1])
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
    # # Decompose the Mi tensor spanning sites i and i+1 with SVD.
    Mi = reshape(Mi, size(psi.M[i], 1), psi.d, psi.d, size(psi.M[i+1], 3))
    dimcutoff = bond_dimension_with_m(psi.L, i+1, m, psi.d)
    U, Vt = sense == 1 ? factorize_svd_right(Mi, dimcutoff=dimcutoff) :
                         factorize_svd_left(Mi, dimcutoff=dimcutoff)

    # Update environments and state.
    Le[i+1] = prop_right3(Le[i], U, H.W[i], U)
    Re[i] = prop_left3(Vt, H.W[i+1], Vt, Re[i+1])
    psi.M[i] = U
    psi.M[i+1] = Vt
    return
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
        array_E, Mi = eigs(Hermitian(Hi), nev=1, which=:SR, v0=v0)
        E = real(array_E[1])
        Mi = vec(Mi)

        # Update left and right environments.
        update_lr_envs_2s!(psi, i, Mi, H, Le, Re, m, sense)

        # Useful debug information.
        if debug > 1
            println("Site: $i, size(Hi): $(size(Hi)), Ei: $(E)")
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
                       m::Int, α::Float64, sense::Int) where T<:Number

Update the left and right environments after the local Hamiltonian is minimized.
"""
function update_lr_envs_3s!(psi::Mps{T}, i::Int, Mi::Vector{T}, H::Mpo{T},
                            Le::Vector{Array{T, 3}}, Re::Vector{Array{T, 3}},
                            m::Int, α::Float64, sense::Int) where T<:Number
    if sense == +1
        # Subspace expansion.
        Mi = reshape(Mi, (size(Le[i], 1), psi.d, size(Re[i], 1)))
        P = prop_right_subexp(Le[i], H.W[i], Mi)
        P = reshape(P, (size(Le[i], 1)*psi.d, size(Re[i], 2)*size(Re[i], 3)))
        # Add subpsace expansion to local mimimum.
        Mi = reshape(Mi, (size(Le[i], 1)*psi.d, size(Re[i], 1)))
        MP = hcat(Mi, α*P)

        # Svd.
        Ai, SV = factorize_svd_right(
            reshape(MP, 1, size(MP)...),
            cutoff=0.,
            dimcutoff=bond_dimension_with_m(psi.L, i+1, m, psi.d)
        )
        # Trim the dimension of SV to match with psi.M[i+1].
        SV = SV[:, 1:size(Re[i], 1)]
        # Unbind the bond dimension in Ai.
        Ai = reshape(Ai, size(Le[i], 1), psi.d, size(Ai, 3))

        # Update left and right environments at Le[i+1] and Re[i] and psi.
        # Ai = reshape(U, (size(Le[i], 1), psi.d, new_m))
        psi.M[i] = Ai
        if i < psi.L
            Le[i+1] = prop_right3(Le[i], Ai, H.W[i], Ai)
            # Absorb SV into psi.M[i+1].
            Ci = reshape(psi.M[i+1], (size(psi.M[i+1], 1), psi.d*size(psi.M[i+1], 3)))
            Ci = SV*Ci
            psi.M[i+1] = reshape(Ci, (size(SV, 1), psi.d, size(Re[i+1], 1)))
            Re[i] = absorb_Re(Re[i], SV)
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
        US, Bi = factorize_svd_left(
            reshape(MP, size(MP)..., 1),
            cutoff=0.,
            dimcutoff=bond_dimension_with_m(psi.L, i, m, psi.d)
        )
        # Trim the dimension of US to match with psi.M[i-1].
        US = US[1:size(Le[i], 1), :]
        # Unbind the bond dimension in Bi.
        Bi = reshape(Bi, size(Bi, 1), psi.d, size(Re[i], 1))

        # Update left and right environments at Le[i+1] and Re[i] and psi.
        psi.M[i] = Bi
        if i > 1
            Re[i-1] = prop_left3(Bi, H.W[i], Bi, Re[i])
            # Absorb US into psi.M[i-1].
            Ci = reshape(psi.M[i-1], (size(psi.M[i-1], 1)*psi.d, size(psi.M[i-1], 3)))
            Ci = Ci*US
            psi.M[i-1] = reshape(Ci, (size(Le[i-1], 1), psi.d, size(US, 2)))
            Le[i] = absorb_Le(Le[i], US)
        end
    end
    return psi
end

"""
    do_sweep_3s!(psi::Mps{T}, H::Mpo{T},
                 Le::Vector{Array{T, 3}}, Re::Vector{Array{T, 3}},
                 sense::Int, m::Int, α::Float64,
                 cache::Cache{T}, debug::Int=0) where T<:Number

Do a sweep to locally minimize the energy of `psi` at 1 site per step. The
direction of the sweep is given by `sense = +1, -1`.
"""
function do_sweep_3s!(psi::Mps{T}, H::Mpo{T},
                      Le::Vector{Array{T, 3}}, Re::Vector{Array{T, 3}},
                      sense::Int, m::Int, α::Float64,
                      cache::Cache{T}, debug::Int=0) where T<:Number

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
        array_E, Mi = eigs(Hermitian(Hi), nev=1, which=:SR, v0=v0)
        E1 = real(array_E[1])
        delta_E1 = E1-E
        Mi = vec(Mi)

        # Update left and right environments.
        update_lr_envs_3s!(psi, i, Mi, H, Le, Re, m, α, sense)

        # Compute new energy and update α.
        E = sense == +1 ? sum(real(Re[i].*Le[i+1])) : sum(real(Re[i-1].*Le[i]))
        delta_E = E-E1
        if -delta_E/delta_E1 > 0.3/log(m)
            α /= 2.
        elseif 0. < -delta_E/delta_E1 < min(1/m, 0.1)
            α *= 2.
        end

        # Useful debug information.
        debug > 1 && println("site: $i, size(Hi): $(size(Hi)), Ei: $(E), α: $(α)")
        debug > 2 && @printf("ΔE_0: %.3e, ΔE_0/ΔE_T: %.3e\n", delta_E1,
                             delta_E/delta_E1)
    end
    return E, α
end
