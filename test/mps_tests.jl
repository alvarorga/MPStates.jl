"""Test whether a tensor product is left canonical."""
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

"""Test whether a tensor product is right canonical."""
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

@testset "Initialization of the Mps class" begin
@testset "make tensor state left & right canonical" begin
    # Test if state is initialized in left canonical form if it is normalized.
    rtest1 = MPStates.testMps("rtest1")
    @test is_left_canonical(rtest1.M)
    @test MPStates.norm(rtest1) ≈ 1.
    MPStates.make_right_canonical!(rtest1)
    @test is_right_canonical(rtest1.M)
    @test MPStates.norm(rtest1) ≈ 1.
end
end # @testset "Initialization of the Mps class"
