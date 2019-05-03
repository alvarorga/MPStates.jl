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
"""
mutable struct Mpo{T}
    M::Vector{Array{T, 4}}
    L::Int
    d::Int
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

    return mpo = Mpo(M, L, 2)
end

"""
    init_mpo(L::Int, J::Array{T, 2}, V::Array{T, 2}, is_fermionic::Bool) where T<:Number

Initialize and Mpo with bond dimension `d=2` with hopping matrix `J` and
an interaction matrix `V = \\sum V_{ij} n_i n_j`. The statistics can be either
fermionic or bosonic.
"""
function init_mpo(L::Int, J::Array{T, 2}, V::Array{T, 2}, is_fermionic::Bool) where T<:Number
    size(J) == (L, L) || throw("J has not the correct dimensions.")
    size(V) == (L, L) || throw("V has not the correct dimensions.")

    M = Vector{Array{T, 4}}()
    Id = Matrix{T}(I, 2, 2)
    # 1 - 2ni operator.
    Z = Matrix{T}(I, 2, 2)
    Z[2, 2] = -one(T)

    # Write basic tensors.
    Mi = zeros(T, 2+2L, 2, 2, 2+2L)
    # Initial and end state Id propagators.
    Mi[1, :, :, 1] = Id
    Mi[2, :, :, 2] = Id
    for i=1:L
        push!(M, deepcopy(Mi))
    end

    # Local terms: J[i, i]*n_i.
    for i=1:L
        M[i][1, 2, 2, 2] = J[i, i]
    end

    # Correlations J_ij*c^dagger_i*c_j.
    for i=1:L, j=1:i-1 # i > j.
        abs(J[i, j]) < 1e-8 && continue
        # Operator c_j.
        M[j][1, 1, 2, 2+i] = J[i, j]
        # Operator Id for bosons or 1-2n for fermions.
        for k=j+1:i-1
            if is_fermionic
                M[k][2+i, :, :, 2+i] = Z
            else
                M[k][2+i, :, :, 2+i] = Id
            end
        end
        # Operator c^dagger_i.
        M[i][2+i, 2, 1, 2] = one(T)
    end
    for i=1:L, j=i+1:L # i < j.
        abs(J[i, j]) < 1e-8 && continue
        # Operator c^dagger_i.
        M[i][1, 2, 1, 2+i] = one(T)
        # Operator Id for bosons or 1-2n for fermions.
        for k=i+1:j-1
            if is_fermionic
                M[k][2+i, :, :, 2+i] = Z
            else
                M[k][2+i, :, :, 2+i] = Id
            end
        end
        # Operator c_j.
        M[j][2+i, 1, 2, 2] = J[i, j]
    end

    # Interactions V_ij*n_i*n_j.
    for j=2:L, i=1:j-1 # i < j.
        abs(V[i, j] + V[j, i]) < 1e-8 && continue
        # Operator n_i.
        M[i][1, 2, 2, 2+L+i] = one(T)
        # Operator Id.
        for k=i+1:j-1
            M[k][2+L+i, :, :, 2+L+i] = Id
        end
        # Operator n_j.
        M[j][2+L+i, 2, 2, 2] = V[i, j] + V[j, i]
    end

    M[1] = M[1][1:1, :, :, :]
    M[end] = M[end][:, :, :, 2:2]

    return Mpo(M, L, 2)
end
