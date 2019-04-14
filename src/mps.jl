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
- `name::String`: name of the Mps. Examples: "GHZ", "W", "product".
- `d::Int=2`: physical dimension.
"""
function init_mps(dtype::Type, L::Int, name::String, d::Int=2)
    A = Vector{Array{Float64, 3}}()
    B = Vector{Array{Float64, 3}}()
    d = 2
    D = ones(Int, L+1)
    for i=1:L
        push!(A, rand(1, 2, 2))
        push!(B, rand(1, 2, 2))
        D[i+1] = size(A[i], 3)
    end
    mps = Mps(A, B, L, d, D)
end
