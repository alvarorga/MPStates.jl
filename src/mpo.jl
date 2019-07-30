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
    init_mpo(T::Type, L::Int, d::Int)

Initialize an Mpo of type `T` and length `L` with bond dimension `d` as an empty
operator.
"""
function init_mpo(T::Type, L::Int, d::Int)
    Id = Matrix{T}(I, d, d)
    W = Vector{Array{T, 4}}(undef, L)
    W1 = zeros(T, 1, d, d, 2)
    W1[1, :, :, 1] = Id
    W[1] = W1
    Wi = zeros(T, 2, d, d, 2)
    Wi[1, :, :, 1] = Id
    Wi[2, :, :, 2] = Id
    for i=2:L-1
        W[i] = deepcopy(Wi)
    end
    Wend = zeros(T, 2, d, d, 1)
    Wend[2, :, :, 1] = Id
    W[end] = Wend
    return Mpo{T}(W, L, d)
end

"""
    add_ops!(Op::Mpo{T}, op_i::AbstractMatrix{<:Number},
             weights::Vector{T}) where T<:Number

Add local/on-site operators to an Mpo.

New operator is: Op + ∑ weights[i]*op_i.
"""
function add_ops!(Op::Mpo{T}, op_i::AbstractMatrix{<:Number},
                  weights::Vector{T}) where T<:Number
    # Convert op_i to have the same type as Op.
    c_op_i = convert.(T, op_i)

    for i=1:Op.L-1
        Op.W[i][1, :, :, 2] .+= weights[i]*c_op_i
    end
    Op.W[end][1, :, :, 1] .+= weights[end]*c_op_i
    return Op
end

"""
    add_ops!(Op::Mpo{T}, str_op_i::String,
             weights::Vector{T}) where T<:Number

Add local/on-site operators to an Mpo.

New operator is: Op + ∑ weights[i]*op_i.
"""
function add_ops!(Op::Mpo{T}, str_op_i::String,
                  weights::Vector{T}) where T<:Number
    return add_ops!(Op, str_to_op(str_op_i), weights)
end

"""
    add_ops!(Op::Mpo{T}, op_i::AbstractMatrix{<:Number},
             op_j::AbstractMatrix{<:Number},
             weights::AbstractMatrix{T};
             ferm_op::AbstractMatrix{<:Number}=Matrix{Float64}(I, Op.d, Op.d)
             ) where T<:Number

Add operators to an Mpo acting on two sites.

New operator is: Op + ∑ weights[i, j]*op_i*op_j. `ferm_op` is the fermion parity
operator: 1-2n that goes between the fermionic creation and annihilation
operators. If the operators `op_i`, `op_j` are not fermionic, `ferm_op` defaults
to the identity matrix.
"""
function add_ops!(Op::Mpo{T}, op_i::AbstractMatrix{<:Number},
                  op_j::AbstractMatrix{<:Number},
                  weights::AbstractMatrix{T};
                  ferm_op::AbstractMatrix{<:Number}=Matrix{Float64}(I, Op.d, Op.d)
                  ) where T<:Number
    # Extract on-site operators.
    onsite_weights = diag(weights)
    add_ops!(Op, op_j*op_i, onsite_weights)

    # Convert op_i, op_j to have the same type as Op.
    c_op_i = convert.(T, op_i)
    c_op_j = convert.(T, op_j)

    # Current bond dimensions of the Mpo.
    max_w = maximum(size.(Op.W, 4))
    L = Op.L

    # Allocate space for the new operators. Let the operators at i=1 and i=L
    # have dimensions (1, d, d, max_w) and (max_w, d, d, 1).
    for i=1:L
        Wi = Op.W[i]
        tmp_wi = zeros(eltype(Wi), i > 1 ? max_w + L : 1,
                                   size(Wi, 2),
                                   size(Wi, 3),
                                   i < L ? max_w + L : 1)
        tmp_wi[1:size(Wi, 1), 1:size(Wi, 2), 1:size(Wi, 3), 1:size(Wi, 4)] += Wi
        Op.W[i] = tmp_wi
    end

    # Write the new operators. Each iteration corresponds to writing all
    # operators in weights[i, :]*op_i*op_j.
    for i=1:L
        # Continue if there are no operators to write.
        all(abs.(weights[i, :]) .< 1e-8) && continue

        # Write op_i.
        if i > 1
            Op.W[i][max_w+i, :, :, i < L ? 2 : 1] = op_i
        end
        if i < L
            Op.W[i][1, :, :, max_w+i] = op_i
        end

        # Write J[i, j]*op_j, j != i.
        for j=1:i-1
            Op.W[j][1, :, :, max_w+i] = op_j*weights[i, j]
        end
        for j=i+1:L-1
            Op.W[j][max_w+i, :, :, 2] = op_j*weights[i, j]
        end
        Op.W[L][max_w+i, :, :, 1] += op_j*weights[i, L]

        # Write operators between op_i and op_j.
        if any(abs.(weights[i, 1:i-1]) .> 1e-8)
            for j=findfirst(abs.(weights[i, :]) .> 1e-8)+1:i-1
                Op.W[j][max_w+i, :, :, max_w+i] = ferm_op
            end
        end
        if any(abs.(weights[i, i+1:L]) .> 1e-8)
            for j=i+1:findlast(abs.(weights[i, :]) .> 1e-8)-1
                Op.W[j][max_w+i, :, :, max_w+i] = ferm_op
            end
        end
    end

    # TODO: compress the operators of the Mpo to remove unnecessary zeros.

    return Op
end

"""
    add_ops!(Op::Mpo{T}, op_i::String, op_j::String,
             weights::AbstractMatrix{T};
             ferm_op::String="Id") where T<:Number

Add operators to an Mpo acting on two sites.

New operator is: Op + ∑ weights[i, j]*op_i*op_j.
"""
function add_ops!(Op::Mpo{T}, op_i::String, op_j::String,
                  weights::AbstractMatrix{T};
                  ferm_op::String="Id") where T<:Number
    return add_ops!(Op, str_to_op(op_i), str_to_op(op_j), weights,
                    ferm_op=str_to_op(ferm_op, Op.d))
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
