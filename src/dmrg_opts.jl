export MinimizeOpts

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
