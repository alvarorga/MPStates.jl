export DMRGOpts

"""
Options for the DMRG minimization algorithm.

Fields:
    algorithm::String: DMRG algorithm used for minimizing the state. Can be:
        "DMRG1" for one site DMRG, "DMRG2" for two site DMRG, and "DMRG3S" for
        strictly single site DMRG with subspace expansion.
    nsweeps::Int: number of sweeps.
    maxm::Vector{Int}: maximum bond dimension at every sweep.
    cutoff::Vector{Float64}: cutoff in SVD factorization at every sweep.
    dmrg_tol::Float64: stop the algorithm when the change in the variance of the
        state is less than dmrg_tol.
    show_trace::Int: output information at every step of the minmization:
        - 0: no info given.
        - 1: energy, variance and their variation with respect to their last
            value after every right + left sweep.
        - 2: energy and size of the local Hamiltonian at every step of every
            sweep.
    lanczos_iters::Int: number of maximum iterations in Arpack solver for
        minimizing local Hamiltonians.
    α::Float64: noise term in DMRG3S.
"""
struct DMRGOpts
    algorithm::String
    nsweeps::Int
    maxm::Vector{Int}
    cutoff::Vector{Float64}
    dmrg_tol::Float64
    show_trace::Int
    lanczos_iters::Int
    α::Float64
end

"""
    DMRGOpts(algorithm::String,
             maxm::Vector{Int},
             cutoff::Vector{Float64};
             dmrg_tol::Float64=1e-6,
             show_trace::Int=1,
             lanczos_iters::Int=6,
             α::Float64=1e-6)

Make DMRG options.
"""
function DMRGOpts(algorithm::String,
                  maxm::Vector{Int},
                  cutoff::Vector{Float64};
                  dmrg_tol::Float64=1e-6,
                  show_trace::Int=1,
                  lanczos_iters::Int=6,
                  α::Float64=1e-6)
    length(maxm) != length(cutoff) && throw(
        DimensionMismatch("maxm and cutoff have different dimensions")
    )
    return DMRGOpts(algorithm, length(maxm), maxm, cutoff,
                    dmrg_tol, show_trace, lanczos_iters, α)
end

"""
    DMRGOpts(algorithm::String,
             maxm::Vector{Int},
             cutoff::Float64;
             kwargs...)

Make DMRG options.
"""
function DMRGOpts(algorithm::String,
                  maxm::Vector{Int},
                  cutoff::Float64;
                  kwargs...)
    return DMRGOpts(algorithm, maxm, fill(cutoff, length(maxm)); kwargs...)
end

"""
    DMRGOpts(algorithm::String,
             maxm::Vector{Int};
             kwargs...)

Make DMRG options.
"""
function DMRGOpts(algorithm::String,
                  maxm::Vector{Int};
                  kwargs...)
    cutoff = get(kwargs, :cutoff, 1e-8)
    return DMRGOpts(algorithm, maxm, cutoff; kwargs...)
end

import Base.display

function Base.display(dmrg_opts::DMRGOpts)
    println("DMRG Options")
    println("  Algorithm: $(dmrg_opts.algorithm)")
    println("  Number of sweeps: $(dmrg_opts.nsweeps)")
    println("  DMRG tolerance: $(dmrg_opts.dmrg_tol)")
    println("  Show trace: $(dmrg_opts.show_trace)")
    println("  Lanczos iterations: $(dmrg_opts.lanczos_iters)")
    println("  maxm   cutoff")
    for i in eachindex(dmrg_opts.maxm)
        println("  $(dmrg_opts.maxm[i])     $(dmrg_opts.cutoff[i])")
    end
end
