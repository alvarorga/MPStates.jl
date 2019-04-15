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
    @test isapprox(norm(leftcan_A[end]), 1.)
    @test isapprox(norm(rightcan_A[1]), 1.)
end
