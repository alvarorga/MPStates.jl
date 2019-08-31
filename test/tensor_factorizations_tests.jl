using Test, LinearAlgebra, MPStates

@testset "Tensor factorizations" begin
@testset "QR decomposition" begin
    (m1, m2, m3) = (4, 5, 6)
    M = randn(m1, m2, m3)
    Q, R = factorize_qr(M)

    # Check dims.
    @test size(Q) == (4, 5, 6)
    @test size(R) == (6, 6)

    # Check Q*R=M.
    Q = reshape(Q, 4*5, 6)
    @test Q*R ≈ reshape(M, 4*5, 6)

    # Check that Q is unitary.
    @test Q'*Q ≈ Matrix(I, 6, 6)

end

@testset "LQ decomposition" begin
    (m1, m2, m3) = (4, 5, 6)
    M = randn(m1, m2, m3)
    L, Q = factorize_lq(M)

    # Check dims.
    @test size(L) == (4, 4)
    @test size(Q) == (4, 5, 6)

    # Check L*Q=M.
    Q = reshape(Q, 4, 5*6)
    @test L*Q ≈ reshape(M, 4, 5*6)

    # Check that Q is unitary.
    @test Q*Q' ≈ Matrix(I, 4, 4)
end
end # @testset "Tensor factorizations"
