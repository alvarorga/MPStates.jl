export build_xxz_mpo,
       build_ssh_mpo

"""
    build_xxz_mpo(::Type{T},
                  L::Int,
                  Δ::Float64,
                  has_pbc::Bool=false) where {T<:Number}

Build the XXZ Hamiltonian:
    H = ∑_i (s⁺_i s⁻_{i+1} + h.c.) + Δ ∑_i sᶻ_i sᶻ_{i+1}
"""
function build_xxz_mpo(::Type{T},
                       L::Int,
                       Δ::Float64,
                       has_pbc::Bool=false) where {T<:Number}
    # S+ S- terms.
    J = diagm(1 => ones(L-1))
    J[1, L] = has_pbc ? 1. : 0.
    J = Symmetric(J)
    # Sz Sz terms.
    V = diagm(1 => fill(Δ, L-1))
    V[1, L] = has_pbc ? Δ : 0.

    H = Mpo(T, L, 2)
    add_ops!(H, "s+", "s-", convert.(T, J))
    add_ops!(H, "sz", "sz", convert.(T, V))

    return H
end

build_xxz_mpo(L, Δ, has_pbc=false) = build_xxz_mpo(Float64, L, Δ, has_pbc)

"""
    build_ssh_mpo(::Type{T},
                  L::Int,
                  t::Float64,
                  ε::Float64,
                  w::Vector{Float64},
                  has_pbc::Bool=false) where {T<:Number}

Build the SSH Hamiltonian:
    H = t/2 ∑_i [(1 + ε*(-)^i)*c⁺_i c_{i+1} + h.c.] + ∑_i w_i n_i
"""
function build_ssh_mpo(::Type{T},
                       L::Int,
                       t::Float64,
                       ε::Float64,
                       w::Vector{Float64},
                       has_pbc::Bool=false) where {T<:Number}
    J = zeros(T, L, L)
    for i=1:L
        J[i, i] = w[i]
        i == L && continue
        J[i, i+1] = t/2*(1 + (iseven(i) ? ε : -ε))
    end
    J[L, 1] = has_pbc ? t/2*(1 + (iseven(L) ? ε : -ε)) : 0.
    J = Symmetric(J)

    H = Mpo(T, L, 2)
    add_ops!(H, "c+", "c", J, ferm_op="Z")

    return H
end

build_ssh_mpo(L, t, ε, w, has_pbc=false) = build_ssh_mpo(Float64, L, t, ε, w, has_pbc)
