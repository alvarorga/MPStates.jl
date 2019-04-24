#
# MPS structure.
#


"""
    Mps{T}

Matrix product state with type T.

# Attributes:
- `A::Vector{Array{T, 3}}`: left-canonical representation.
- `B::Vector{Array{T, 3}}`: right-canonical representation.
- `L::Int`: length of the Mps.
- `d::Int`: physical bond dimension.
- `D::Vector{Int}`: bond dimension along the sites of the Mps.
"""
mutable struct Mps{T}
    A::Vector{Array{T, 3}}
    B::Vector{Array{T, 3}}
    L::Int
    d::Int
    D::Vector{Int}
end

"""
    init_mps(T::Type, L::Int, name::String, d::Int=2)

Initialize an Mps.

# Arguments:
- `T::Type`: type of the Mps.
- `L::Int`: Mps's length.
- `name::String`: name of the Mps. Examples: "GHZ", "W", "product", "random",
    "full".
- `d::Int=2`: physical dimension.
"""
function init_mps(T::Type, L::Int, name::String, d::Int=2)
    A = Vector{Array{T, 3}}()
    B = Vector{Array{T, 3}}()
    d = 2
    D = ones(Int, L+1)
    # Build basis for constructing states.
    if name == "product"
        M1 = zeros(T, 1, d, 1)
        M1[1, 1, 1] = one(T)
        M = copy(M1)
        Mend = copy(M1)
    elseif name == "GHZ"
        M1 = zeros(T, 1, 2, 2)
        M1[1, 1, 1] = one(T)
        M1[1, 2, 2] = one(T)
        M = zeros(T, 2, 2, 2)
        M[1, 1, 1] = one(T)
        M[2, 2, 2] = one(T)
        Mend = zeros(T, 2, 2, 1)
        Mend[1, 1, 1] = one(T)
        Mend[2, 2, 1] = one(T)
    elseif name == "W"
        M1 = zeros(T, 1, 2, 2)
        M1[1, 1, 1] = one(T)
        M1[1, 2, 2] = one(T)
        M = zeros(T, 2, 2, 2)
        M[1, 1, 1] = one(T)
        M[2, 1, 2] = one(T)
        M[1, 2, 2] = one(T)
        Mend = zeros(T, 2, 2, 1)
        Mend[2, 1, 1] = one(T)
        Mend[1, 2, 1] = one(T)
    elseif name == "random"
        M1 = rand(T, 1, d, 2)
        M = rand(T, 2, d, 2)
        Mend = rand(T, 2, d, 1)
    elseif name == "full"
        M1 = ones(T, 1, d, 1)
        M = ones(T, 1, d, 1)
        Mend = ones(T, 1, d, 1)
    end

    # Join all tensors in arrays and make left- and right-canonical.
    push!(A, M1)
    push!(B, M1)
    D[2] = size(A[1], 3)
    for i=2:L-1
        push!(A, M)
        push!(B, M)
        D[i+1] = size(A[i], 3)
    end
    push!(A, Mend)
    push!(B, Mend)
    D[L+1] = size(A[L], 3)
    A = make_left_canonical(A)
    B = make_right_canonical(B)

    mps = Mps(A, B, L, d, D)
end

"""
    make_left_canonical(A::Vector{Array{T, 3}}) where T

Take a vector of tensors and make it left-canonical.
"""
function make_left_canonical(A::Vector{Array{T, 3}}) where T<:Number
    leftcan_A = Vector{Array{T, 3}}()
    R = ones(T, 1, 1)
    for i=1:length(A)-1
        Q, R = prop_qr(R, A[i])
        push!(leftcan_A, Q)
    end
    # Normalize and append last tensor.
    @tensor Aend[i, s, j] := R[i, k]*A[end][k, s, j]
    Aend ./= norm(Aend)
    push!(leftcan_A, Aend)
    return leftcan_A
end

"""
    make_right_canonical(A::Vector{Array{T, 3}}) where T

Take a vector of tensors and make it right-canonical.
"""
function make_right_canonical(A::Vector{Array{T, 3}}) where T<:Number
    rightcan_A = Vector{Array{T, 3}}()
    L = ones(T, 1, 1)
    for i=length(A):-1:2
        L, Q = prop_lq(A[i], L)
        push!(rightcan_A, Q)
    end
    # Normalize and append last tensor.
    @tensor Aend[i, s, j] := A[1][i, s, k]*L[k, j]
    Aend ./= norm(Aend)
    push!(rightcan_A, Aend)
    return reverse(rightcan_A)
end
