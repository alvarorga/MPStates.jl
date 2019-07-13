#
# MPO structure.
#


"""
    Mpo{T<:Number}

Matrix product operator with type T.

# Attributes:
- `W::Vector{Array{T, 4}}`: tensor representation of the Mpo.
- `L::Int`: length of the Mpo.
- `d::Int`: physical bond dimension.
"""
struct Mpo{T<:Number}
    W::Vector{Array{T, 4}}
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
    W = Vector{Array{T, 4}}()
    # Tensors that make the structure of the Mpo.
    Wi = zeros(T, 5, 2, 2, 5)
    # Identity operators.
    Id = Matrix{T}(I, 2, 2)
    Wi[1, :, :, 1] = Id
    Wi[2, :, :, 2] = Id
    # Creation operators.
    Wi[1, 1, 2, 3] = convert(T, t)
    Wi[4, 1, 2, 2] = 1.
    # Annihilation operators.
    Wi[3, 2, 1, 2] = 1.
    Wi[1, 2, 1, 4] = convert(T, t)
    # Number operators.
    Wi[1, 2, 2, 5] = convert(T, U)
    Wi[5, 2, 2, 2] = 1.
    # Initial and final tensors.
    W0 = reshape(Wi[1, :, :, :], (1, size(Wi, 2), size(Wi, 3), size(Wi, 4)))
    W_end = reshape(Wi[:, :, :, 2], (size(Wi, 1), size(Wi, 2), size(Wi, 3), 1))

    push!(W, W0)
    for i=2:L-1
        push!(W, Wi)
    end
    push!(W, W_end)

    return Mpo(W, L, 2)
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

    W = Vector{Array{T, 4}}()
    Id = Matrix{T}(I, 2, 2)
    # 1 - 2ni operator.
    Z = Matrix{T}(I, 2, 2)
    Z[2, 2] = -one(T)

    # Write basic tensors.
    Wi = zeros(T, 2+2L, 2, 2, 2+2L)
    # Initial and end state Id propagators.
    Wi[1, :, :, 1] = Id
    Wi[2, :, :, 2] = Id
    for i=1:L
        push!(W, deepcopy(Wi))
    end

    # Local terms: J[i, i]*n_i.
    for i=1:L
        W[i][1, 2, 2, 2] = J[i, i]
    end

    # Keep trace of final occupied indices in Mpo.
    occ_ix = zeros(Int, 2+2L)
    occ_ix[1] = L
    occ_ix[2] = L

    # Correlations J_ij*c^dagger_i*c_j.
    for i=1:L
        norm(J[i, :]) < 1e-8 && continue
        ix_initial = min(i, findfirst(abs.(J[i, :]) .> 1e-8))
        ix_final = max(i, findlast(abs.(J[i, :]) .> 1e-8))
        ix = findfirst(ix_initial .>= occ_ix)
        occ_ix[ix] = ix_final

        for j=1:i-1 # i > j.
            abs(J[i, j]) <= 1e-8 && continue
            # Operator c_j.
            W[j][1, 2, 1, ix] = J[i, j]
            # Operator Id for bosons or 1-2n for fermions.
            for k=j+1:i-1
                if is_fermionic
                    W[k][ix, :, :, ix] = Z
                else
                    W[k][ix, :, :, ix] = Id
                end
            end
            # Operator c^dagger_i.
            W[i][ix, 1, 2, 2] = one(T)
        end
        for j=i+1:L # i < j.
            abs(J[i, j]) < 1e-8 && continue
            # Operator c^dagger_i.
            W[i][1, 1, 2, ix] = one(T)
            # Operator Id for bosons or 1-2n for fermions.
            for k=i+1:j-1
                if is_fermionic
                    W[k][ix, :, :, ix] = Z
                else
                    W[k][ix, :, :, ix] = Id
                end
            end
            # Operator c_j.
            W[j][ix, 2, 1, 2] = J[i, j]
        end
    end

    # Interactions V_ij*n_i*n_j.
    for i=2:L
        norm(V[i, :] .+ V[:, i]) < 1e-8 && continue
        ix_initial = min(i, findfirst(abs.(V[i, :] .+ V[:, i]) .>= 1e-8))
        ix_final = max(i, findlast(abs.(V[i, :] .+ V[:, i]) .>= 1e-8))
        ix = findfirst(ix_initial .>= occ_ix)
        occ_ix[ix] = ix_final

        for j=1:i-1 # j < i.
            abs(V[i, j] + V[j, i]) < 1e-8 && continue
            # Operator n_j.
            W[j][1, 2, 2, ix] = V[i, j] + V[j, i]
            # Operator Id.
            for k=j+1:i-1
                W[k][ix, :, :, ix] = Id
            end
            # Operator n_i.
            W[i][ix, 2, 2, 2] = one(T)
        end
    end

    # Trim the unfilled region of the W tensors.
    ix_trim = findlast(occ_ix .> 0)
    for i=1:L
        W[i] = W[i][1:ix_trim, :, :, 1:ix_trim]
    end

    W[1] = W[1][1:1, :, :, :]
    W[end] = W[end][:, :, :, 2:2]

    return Mpo(W, L, 2)
end

function Base.display(O::Mpo{T}) where T<:Number
    println("MPO:")
    println("   Type: $(eltype(O.W[1]))")
    println("   Length: $(O.L)")
    println("   Physical dims: $(O.d)")
    println("   Max bond dim: $(maximum(size.(O.W, 4)))")
    return
end

function show_bond_dims(O::Mpo{T}) where T<:Number
    bdims = vcat(1, size.(O.W[:], 4))
    println(join(bdims, "-"))
    return
end
