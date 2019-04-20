#
# MPO structure.
#


"""
    Mpo{T}

Matrix product operator with type T.

# Attributes:
- `M::Vector{Array{T, 4}}`: tensor that make the Mpo.
- `L::Int`: length of the Mpo.
- `d::Int`: physical bond dimension.
- `is_fermionic::Bool`: whether the Mpo has fermionic statistics or not.
"""
mutable struct Mpo{T}
    M::Vector{Array{T, 4}}
    L::Int
    d::Int
    is_fermionic::Bool
end

"""
    init_hubbard_mpo(L::Int, t::T, U::T) where T<:Number

Initialize a simple Hubbard model as an Mpo:
``H = \\sum t\\ c^\\dagger_i c_{i+1} + U n_i n_{i+1}``

# Arguments:
- `L::Int`: Mpo's length.
- `t::T`: tunneling amplitude
- `U::T`: Hubbard interaction strength.
"""
function init_hubbard_mpo(L::Int, t::T, U::T) where T<:Number
    M = Vector{Array{T, 4}}()
    # Tensors that make the structure of the Mpo.
    Mi = zeros(T, 5, 2, 2, 5)
    # Identity operators.
    Id = Matrix{T}(I, 2, 2)
    Mi[1, :, :, 1] = Id
    Mi[2, :, :, 2] = Id
    # Creation operators.
    Mi[1, 2, 1, 3] = convert(T, t)
    Mi[4, 2, 1, 2] = one(T)
    # Annihilation operators.
    Mi[3, 1, 2, 2] = one(T)
    Mi[1, 1, 2, 4] = convert(T, t)
    # Number operators.
    Mi[1, 2, 2, 5] = convert(T, U)
    Mi[5, 2, 2, 2] = one(T)
    # Initial and final tensors.
    M0 = reshape(Mi[1, :, :, :], (1, size(Mi, 2), size(Mi, 3), size(Mi, 4)))
    Mend = reshape(Mi[:, :, :, 2], (size(Mi, 1), size(Mi, 2), size(Mi, 3), 1))

    push!(M, M0)
    for i=2:L-1
        push!(M, Mi)
    end
    push!(M, Mend)

    mpo = Mpo(M, L, 2, true)
end
