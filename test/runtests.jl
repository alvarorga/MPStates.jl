using Test, LinearAlgebra, MPStates

@testset "make vector of arrays left & right canonical" begin
    L = 5

    # Make vector of rank-3 tensors.
    A = Vector{Array{ComplexF64, 3}}()
    push!(A, rand(ComplexF64, 1, 2, 2))
    for i=2:L-1
        push!(A, rand(ComplexF64, 2, 2, 2))
    end
    push!(A, rand(ComplexF64, 2, 2, 1))

    function is_left_canonical(A::Vector{Array{T, 3}}) where T
        output = true
        for i=1:length(A)-1
            Ai = A[i]
            rA = reshape(Ai, (size(Ai, 1)*size(Ai, 2), size(Ai, 3)))
            M = rA'*rA
            Id = Matrix{T}(I, size(Ai, 3), size(Ai, 3))
            output = isapprox(M, Id)
        end
        return output
    end

    function is_right_canonical(A::Vector{Array{T, 3}}) where T
        output = true
        for i=length(A):-1:2
            Ai = A[i]
            rA = reshape(Ai, (size(Ai, 1), size(Ai, 2)*size(Ai, 3)))
            M = rA*rA'
            Id = Matrix{T}(I, size(Ai, 1), size(Ai, 1))
            output = isapprox(M, Id)
        end
        return output
    end

    leftcan_A = MPStates.make_left_canonical(A)
    rightcan_A = MPStates.make_right_canonical(A)

    @test is_left_canonical(leftcan_A)
    @test is_right_canonical(rightcan_A)

    # Test norm of canonical tensors.
    @test norm(leftcan_A[end]) ≈ 1.
    @test norm(rightcan_A[1]) ≈ 1.
end

@testset "measure occupation at one site" begin
    L = 5
    GHZ = init_mps(Float64, L, "GHZ")
    @test m_occupation(GHZ, 1) ≈ 0.5
    @test m_occupation(GHZ, 2) ≈ 0.5
    @test m_occupation(GHZ, 5) ≈ 0.5
    W = init_mps(Float64, L, "W")
    @test m_occupation(W, 1) ≈ 1/L
    @test m_occupation(W, 3) ≈ 1/L
    @test m_occupation(W, 5) ≈ 1/L
end

@testset "contraction of two MPS" begin
    L = 5
    GHZ = init_mps(Float64, L, "GHZ")
    W = init_mps(Float64, L, "W")
    full = init_mps(Float64, L, "full")
    product = init_mps(Float64, L, "product")
    @test contract(GHZ, W) ≈ 0.
    @test contract(GHZ, full) ≈ 1/sqrt(2^(L-1))
    @test contract(GHZ, product) ≈ 1/sqrt(2)
    @test contract(W, full) ≈ sqrt(L/2^L)
    @test contract(W, product) ≈ 0.
    @test contract(full, product) ≈ 1/sqrt(2^L)
end
