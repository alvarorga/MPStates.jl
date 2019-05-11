#
# MPS structure.
#


"""
    Mps{T<:Number}

Matrix product state with type T.

# Attributes:
- `A::Vector{Array{T, 3}}`: left-canonical representation.
- `B::Vector{Array{T, 3}}`: right-canonical representation.
- `L::Int`: length of the Mps.
- `d::Int`: physical bond dimension.
"""
struct Mps{T<:Number}
    A::Vector{Array{T, 3}}
    B::Vector{Array{T, 3}}
    L::Int
    d::Int
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
    for i=2:L-1
        push!(A, M)
        push!(B, M)
    end
    push!(A, Mend)
    push!(B, Mend)
    A = make_left_canonical(A)
    B = make_right_canonical(B)

    mps = Mps(A, B, L, d)
end

"""
    init_test_mps(name::String)

Initialize an Mps used for testing routines. All test Mps have L=6, d=2.

# Arguments:
- `name::String`: valid are "rtest1", "rtest2", "ctest1", "ctest2" where the
ones starting with "r" are of type Float64 and the other are ComplexF64.

Shape of test states:
- "rtest1": (1/3|100> - 2/3|010> + 2/3|001>)(0.8|111> - 0.6|101>)
- "rtest2": (2/3|01> - 2/3|10> + 1/3|11>)(0.8|1111> + 0.6|1010>)
- "ctest1": (i/3|100> - 2/3|010> + 2i/3|001>)(0.8|111> - 0.6i|101>)
- "ctest2": (2i/3|01> - 2/3|10> + i/3|11>)(0.8i|1111> + 0.6|1010>)
"""
function init_test_mps(name::String)
    if name[1] == 'r'
        T = Float64
    elseif name[1] == 'c'
        T = ComplexF64
    else
        throw("State '$name' not valid.")
    end

    M = Vector{Array{T, 3}}()
    A = Vector{Array{T, 3}}()
    B = Vector{Array{T, 3}}()
    L = 6
    d = 2
    # Build basis for constructing states.
    if name == "rtest1"
        M1 = zeros(T, 1, 2, 2)
        M1[1, 1, 1] = 1.
        M1[1, 2, 2] = 1/3
        M2 = zeros(T, 2, 2, 2)
        M2[1, 1, 1] = 1.
        M2[2, 1, 2] = 1.
        M2[1, 2, 2] = -2/3
        M3 = zeros(T, 2, 2, 2)
        M3[2, 1, 2] = 1.
        M3[1, 2, 2] = 2/3
        M4 = zeros(T, 2, 2, 2)
        M4[2, 2, 2] = 1.
        M5 = zeros(T, 2, 2, 2)
        M5[2, 1, 2] = -0.6
        M5[2, 2, 2] = 0.8
        M6 = zeros(T, 2, 2, 1)
        M6[2, 2, 1] = 1
    elseif name == "rtest2"
        M1 = zeros(T, 1, 2, 2)
        M1[1, 1, 1] = -4/3
        M1[1, 2, 2] = -2/3
        M2 = zeros(T, 2, 2, 2)
        M2[2, 1, 2] = 1.
        M2[1, 2, 1] = -0.5
        M2[2, 2, 2] = -0.5
        M3 = zeros(T, 2, 2, 2)
        M3[1, 2, 1] = 1.
        M3[2, 2, 1] = 1.
        M4 = zeros(T, 2, 2, 2)
        M4[1, 2, 1] = 1.
        M5 = zeros(T, 2, 2, 2)
        M5[1, 2, 1] = 1.
        M6 = zeros(T, 2, 2, 1)
        M6[1, 1, 1] = 0.6
        M6[1, 2, 1] = 0.8
    elseif name == "ctest1"
        M1 = zeros(T, 1, 2, 2)
        M1[1, 1, 1] = 1.
        M1[1, 2, 2] = complex(0., 1/3)
        M2 = zeros(T, 2, 2, 2)
        M2[1, 1, 1] = 1.
        M2[2, 1, 2] = 1.
        M2[1, 2, 2] = -2/3
        M3 = zeros(T, 2, 2, 2)
        M3[2, 1, 2] = 1.
        M3[1, 2, 2] = complex(0., 2/3)
        M4 = zeros(T, 2, 2, 2)
        M4[2, 2, 2] = 1.
        M5 = zeros(T, 2, 2, 2)
        M5[2, 1, 2] = complex(0., -0.6)
        M5[2, 2, 2] = 0.8
        M6 = zeros(T, 2, 2, 1)
        M6[2, 2, 1] = 1
    elseif name == "ctest2"
        M1 = zeros(T, 1, 2, 2)
        M1[1, 1, 1] = -4/3
        M1[1, 2, 2] = -2/3
        M2 = zeros(T, 2, 2, 2)
        M2[2, 1, 2] = 1.
        M2[1, 2, 1] = complex(0., -0.5)
        M2[2, 2, 2] = complex(0., -0.5)
        M3 = zeros(T, 2, 2, 2)
        M3[1, 2, 1] = 1.
        M3[2, 2, 1] = 1.
        M4 = zeros(T, 2, 2, 2)
        M4[1, 2, 1] = 1.
        M5 = zeros(T, 2, 2, 2)
        M5[1, 2, 1] = 1.
        M6 = zeros(T, 2, 2, 1)
        M6[1, 1, 1] = 0.6
        M6[1, 2, 1] = complex(0., 0.8)
    else
        throw("State '$name' not valid.")
    end

    # Join all tensors in arrays and make left- and right-canonical.
    push!(M, M1)
    push!(M, M2)
    push!(M, M3)
    push!(M, M4)
    push!(M, M5)
    push!(M, M6)
    A = make_left_canonical(M)
    B = make_right_canonical(M)

    mps = Mps(A, B, L, d)
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
