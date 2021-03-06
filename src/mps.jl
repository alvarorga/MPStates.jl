export Mps,
       randomMps,
       show_bond_dims,
       max_bond_dim

"""
    Mps{T<:Number}

Matrix product state with type T.

# Attributes:
- `M::Vector{Array{T, 3}}`: tensor representation of the Mps.
- `L::Int`: length of the Mps.
- `d::Int`: physical bond dimension.
"""
struct Mps{T<:Number}
    M::Vector{Array{T, 3}}
    L::Int
    d::Int
end

Mps(L::Int, name::String, d0::Int=2) = Mps(Float64, L, name, d0)
randomMps(L::Int, d::Int=2, m::Int=2) = randomMps(Float64, L, d, m)

"""
    Mps(::Type{T}, L::Int, name::String, d0::Int=2) where T<:Number

Initialize an Mps in left canonical form of the kinds: GHZ, W, product state
(all zero entries), full, and AKLT.

# Arguments:
- `T::Type`: type of the Mps.
- `L::Int`: Mps's length.
- `name::String`: can be: "GHZ", "W", "product", "full", "AKLT".
- `d0::Int=2`: physical dimension.
"""
function Mps(::Type{T}, L::Int, name::String, d0::Int=2) where T<:Number
    M = Vector{Array{T, 3}}()
    d = d0
    # Build basis for constructing states.
    if name == "product"
        M1 = zeros(T, 1, d, 1)
        M1[1, 1, 1] = one(T)
        Mi = copy(M1)
        Mend = copy(M1)
    elseif name == "GHZ"
        M1 = zeros(T, 1, 2, 2)
        M1[1, 1, 1] = one(T)
        M1[1, 2, 2] = one(T)
        Mi = zeros(T, 2, 2, 2)
        Mi[1, 1, 1] = one(T)
        Mi[2, 2, 2] = one(T)
        Mend = zeros(T, 2, 2, 1)
        Mend[1, 1, 1] = one(T)
        Mend[2, 2, 1] = one(T)
    elseif name == "W"
        M1 = zeros(T, 1, 2, 2)
        M1[1, 1, 1] = one(T)
        M1[1, 2, 2] = one(T)
        Mi = zeros(T, 2, 2, 2)
        Mi[1, 1, 1] = one(T)
        Mi[2, 1, 2] = one(T)
        Mi[1, 2, 2] = one(T)
        Mend = zeros(T, 2, 2, 1)
        Mend[2, 1, 1] = one(T)
        Mend[1, 2, 1] = one(T)
    elseif name == "full"
        M1 = ones(T, 1, d, 1)
        Mi = ones(T, 1, d, 1)
        Mend = ones(T, 1, d, 1)
    elseif name == "AKLT"
        d = 3
        M1 = zeros(T, 1, d, 2)
        M1[1, 1, 1] = one(T)/sqrt(3)
        M1[1, 2, 2] = one(T)
        M1[1, 3, 1] = sqrt(2/3)
        Mi = zeros(T, 2, d, 2)
        Mi[2, 1, 1] = sqrt(2/3)
        Mi[1, 2, 1] = -one(T)/sqrt(3)
        Mi[2, 2, 2] = one(T)/sqrt(3)
        Mi[1, 3, 2] = -sqrt(2/3)
        Mend = zeros(T, 2, d, 1)
        Mend[1, 1, 1] = one(T)/sqrt(3)
        Mend[2, 2, 1] = one(T)/sqrt(3)
        Mend[1, 3, 1] = one(T)/sqrt(3)
    end

    # Join all tensors in a vector.
    push!(M, M1)
    for i=2:L-1
        push!(M, Mi)
    end
    push!(M, Mend)

    mps = Mps(M, L, d)
    make_left_canonical!(mps)
    return mps
end

"""
    randomMps(::Type{T}, L::Int, d::Int=2, m::Int=2) where T<:Number

Make a random Mps in left canonical form.

# Arguments:
- `T::Type`: type of the Mps.
- `L::Int`: Mps's length.
- `d::Int=2`: physical dimension.
- `m::Int=2`: maximum bond dimension.
"""
function randomMps(::Type{T}, L::Int, d::Int=2, m::Int=2) where T<:Number
    M = Vector{Array{T, 3}}()

    # Make a vector of random tensors.
    push!(M, rand(T, 1, d, bond_dimension_with_m(L, 2, m, d)))
    for i=2:L-1
        push!(M, rand(T, bond_dimension_with_m(L, i, m, d),
                         d,
                         bond_dimension_with_m(L, i+1, m, d)))
    end
    push!(M, rand(T, bond_dimension_with_m(L, L, m, d), d, 1))

    mps = Mps(M, L, d)
    make_left_canonical!(mps)
    return mps
end

"""
    testMps(name::String)

Make several Mps used for testing routines. All test Mps have L=6.

# Arguments:
- `name::String`: valid are "rtest1", "rtest2", "ctest1", "ctest2" where the
ones starting with "r" are of type Float64 and the other are ComplexF64.

Shape of test states:
- "rtest1": (1/3|100> - 2/3|010> + 2/3|001>)(0.8|111> - 0.6|101>)
- "rtest2": (2/3|01> - 2/3|10> + 1/3|11>)(0.8|1111> + 0.6|1110>)
- "ctest1": (i/3|100> - 2/3|010> + 2i/3|001>)(0.8|111> - 0.6i|101>)
- "ctest2": (2i/3|01> - 2/3|10> + i/3|11>)(0.8i|1111> + 0.6|1110>)
- "rtest3": 1/6*(|000> - |+++> + |--0> - |0+-> + |0++> - |-+0>)
               *(|000> - |+++> + |--0> - |0+-> + |0++> - |-+0>)
"""
function testMps(name::String)
    if name[1] == 'r'
        T = Float64
    elseif name[1] == 'c'
        T = ComplexF64
    else
        throw("State '$name' not valid.")
    end

    M = Vector{Array{T, 3}}()
    L = 6
    d = -1
    # Build basis for constructing states.
    if name == "rtest1"
        d = 2
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
        d = 2
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
        d = 2
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
        d = 2
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
    elseif name == "rtest3"
        d = 3
        M1 = zeros(T, 1, 3, 3)
        M1[1, 1, 1] = sqrt(1/6)
        M1[1, 2, 3] = sqrt(1/6)
        M1[1, 3, 2] = -sqrt(1/6)
        M2 = zeros(T, 3, 3, 3)
        M2[1, 1, 1] = 1.
        M2[3, 2, 1] = 1.
        M2[1, 3, 1] = -1.
        M2[2, 3, 2] = 1.
        M2[3, 3, 3] = 1.
        M3 = zeros(T, 3, 3, 1)
        M3[3, 1, 1] = -1.
        M3[1, 2, 1] = 1.
        M3[2, 3, 1] = 1.
        M3[3, 3, 1] = 1.
        M4 = M1
        M5 = M2
        M6 = M3
    else
        throw("State '$name' not valid.")
    end

    # Join all tensors in a vector.
    push!(M, M1)
    push!(M, M2)
    push!(M, M3)
    push!(M, M4)
    push!(M, M5)
    push!(M, M6)

    mps = Mps(M, L, d)
    make_left_canonical!(mps)
    return mps
end

"""
    make_left_canonical!(psi::Mps{T}, normalize::Bool=true) where T<:Number

Write an Mps in left canonical form.
"""
function make_left_canonical!(psi::Mps{T}, normalize::Bool=true) where T<:Number
    R = ones(T, 1, 1)
    for i=1:psi.L-1
        psi.M[i] = absorb_fromleft(R, psi.M[i])
        psi.M[i], R = factorize_qr(psi.M[i])
    end
    # Last tensor.
    @tensor A_end[i, s, j] := R[i, k]*psi.M[end][k, s, j]
    if normalize
        A_end ./= norm(A_end)
    end
    psi.M[end] = A_end
    return psi
end

"""
    make_right_canonical!(psi::Mps{T}, normalize::Bool=true) where T<:Number

Write an Mps in right canonical form.
"""
function make_right_canonical!(psi::Mps{T}, normalize::Bool=true) where T<:Number
    L = ones(T, 1, 1)
    for i=reverse(2:psi.L)
        psi.M[i] = absorb_fromright(psi.M[i], L)
        L, psi.M[i] = factorize_lq(psi.M[i])
    end
    # Normalize and append last tensor.
    @tensor B_end[i, s, j] := psi.M[1][i, s, k]*L[k, j]
    if normalize
        B_end ./= norm(B_end)
    end
    psi.M[1] = B_end
    return psi
end

"""
    Base.display(psi::Mps{<:Number})

Show relevant information of `psi`.
"""
function Base.display(psi::Mps{<:Number})
    println("MPS:")
    println("   Type: $(eltype(psi.M[1]))")
    println("   Length: $(psi.L)")
    println("   Physical dims: $(psi.d)")
    println("   Max bond dim: $(maximum(size.(psi.M, 3)))")
    return
end

"""
    show_bond_dims(psi::Mps{<:Number})

Show the bond dimensions of `psi`.
"""
function show_bond_dims(psi::Mps{<:Number})
    bdims = vcat(1, size.(psi.M, 3))
    println(join(bdims, "-"))
    return
end

"""
    max_bond_dim(psi::Mps{<:Number})

Maximum bond dimension of `psi`.
"""
function max_bond_dim(psi::Mps{<:Number})
    return maximum(size.(psi.M, 1))
end
