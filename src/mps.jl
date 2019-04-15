#
# MPS structure.
#


"""
    Mps{T}

Matrix product state with type T.

# hAttributes:
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
    init_mps(dtype::Type, L::Int, name::String, d::Int=2)

Initialize an Mps.

# Arguments:
- `dtype::Type`: type of the Mps.
- `L::Int`: Mps's length.
- `name::String`: name of the Mps. Examples: "GHZ", "W", "product", "random".
- `d::Int=2`: physical dimension.
"""
function init_mps(dtype::Type, L::Int, name::String, d::Int=2)
    A = Vector{Array{Float64, 3}}()
    B = Vector{Array{Float64, 3}}()
    d = 2
    D = ones(Int, L+1)
    # Build basis for constructing states.
    if name == "product"
        M1 = zeros(dtype, 1, d, 1)
        M1[1, 1, 1] = one(dtype)
        M = copy(M1)
        Mend = copy(M1)
    elseif name == "GHZ"
        M1 = zeros(dtype, 1, 2, 2)
        M1[1, 1, 1] = one(dtype)
        M1[1, 2, 2] = one(dtype)
        M = zeros(dtype, 2, 2, 2)
        M[1, 1, 1] = one(dtype)
        M[2, 2, 2] = one(dtype)
        Mend = zeros(dtype, 2, 2, 1)
        Mend[1, 1, 1] = one(dtype)
        Mend[2, 2, 1] = one(dtype)
    elseif name == "W"
        M1 = zeros(dtype, 1, 2, 2)
        M1[1, 1, 1] = one(dtype)
        M1[1, 2, 2] = one(dtype)
        M = zeros(dtype, 2, 2, 2)
        M[1, 1, 1] = one(dtype)
        M[2, 1, 2] = one(dtype)
        M[1, 2, 2] = one(dtype)
        Mend = zeros(dtype, 2, 2, 1)
        Mend[2, 1, 1] = one(dtype)
        Mend[1, 2, 1] = one(dtype)
    elseif name == "random"
        M1 = rand(dtype, 1, d, 2)
        M = rand(dtype, 2, d, 2)
        Mend = rand(dtype, 2, d, 1)
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
function make_left_canonical(A::Vector{Array{T, 3}}) where T
    leftcan_A = Vector{Array{T, 3}}()
    Ai = A[1]
    for i=1:length(A)-1
        rA = reshape(Ai, (size(Ai, 1)*size(Ai, 2), size(Ai, 3)))
        rQ, R = qr(rA)
        rQ = Matrix(rQ)
        Q = reshape(rQ, (size(Ai, 1), size(Ai, 2), size(rQ, 2)))
        push!(leftcan_A, Q)

        # Contract R with tensor in site i+1.
        Ai = A[i+1]
        # TODO: join following two lines in one tensordot.
        rAi = R*reshape(Ai, (size(Ai, 1), size(Ai, 2)*size(Ai, 3)))
        Ai = reshape(rAi, (size(R, 1), size(Ai, 2), size(Ai, 3)))
    end
    # Normalize and append last tensor.
    Ai ./= norm(Ai)
    push!(leftcan_A, Ai)
    return leftcan_A
end

"""
    make_right_canonical(A::Vector{Array{T, 3}}) where T

Take a vector of tensors and make it right-canonical.
"""
function make_right_canonical(A::Vector{Array{T, 3}}) where T
    rightcan_A = Vector{Array{T, 3}}()
    Ai = A[end]
    for i=length(A):-1:2
        rA = reshape(Ai, (size(Ai, 1), size(Ai, 2)*size(Ai, 3)))
        L, rQ = lq(rA)
        rQ = Matrix(rQ)
        Q = reshape(rQ, (size(rQ, 1), size(Ai, 2), size(Ai, 3)))
        push!(rightcan_A, Q)

        # Contract R with tensor in site i-1.
        Ai = A[i-1]
        # TODO: join following two lines in one tensordot.
        rAi = reshape(Ai, (size(Ai, 1)*size(Ai, 2), size(Ai, 3)))*L
        Ai = reshape(rAi, (size(Ai, 1), size(Ai, 2), size(L, 2)))
    end
    # Normalize and append last tensor.
    Ai ./= norm(Ai)
    push!(rightcan_A, Ai)
    return reverse(rightcan_A)
end
