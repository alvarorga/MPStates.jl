using Test, Random, LinearAlgebra, MPStates

@testset "Tensor factorizations" begin
Random.seed!(0)

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

@testset "SVD right decomposition" begin
    (m1, m2, m3, m4) = (3, 4, 5, 3)
    M = randn(m1, m2, m3, m4)

    # Full factorization.
    U, SVt = factorize_svd_right(M, cutoff=0., normalize_S=false)
    @test size(U) == (3, 4, 12)
    @test size(SVt) == (12, 5, 3)
    rU = reshape(U, 12, 12)
    @test rU'*rU ≈ Matrix(I, 12, 12)
    rSVt = reshape(SVt, 12, 15)
    @test rU*rSVt ≈ reshape(M, 12, 15)

    # SVD with cutoff.
    U, SVt = factorize_svd_right(M, cutoff=1., normalize_S=false)
    @test size(U, 1) == 3
    @test size(U, 2) == 4
    @test size(U, 3) < 12
    @test size(SVt, 2) == 5
    @test size(SVt, 3) == 3
    @test size(SVt, 1) < 12
    rU = reshape(U, 12, size(U, 3))
    @test rU'*rU ≈ Matrix(I, size(U, 3), size(U, 3))
end

@testset "SVD left decomposition" begin
    (m1, m2, m3, m4) = (3, 4, 5, 3)
    M = randn(m1, m2, m3, m4)

    # Full factorization.
    US, Vt = factorize_svd_left(M, cutoff=0., normalize_S=false)
    @test size(US) == (3, 4, 12)
    @test size(Vt) == (12, 5, 3)
    rUS = reshape(US, 12, 12)
    rVt = reshape(Vt, 12, 15)
    @test rVt*rVt' ≈ Matrix(I, 12, 12)
    @test rUS*rVt ≈ reshape(M, 12, 15)

    # SVD with cutoff.
    US, Vt = factorize_svd_left(M, cutoff=1., normalize_S=false)
    @test size(US, 1) == 3
    @test size(US, 2) == 4
    @test size(US, 3) < 12
    @test size(Vt, 2) == 5
    @test size(Vt, 3) == 3
    @test size(Vt, 1) < 12
    rVt = reshape(Vt, size(Vt, 1), 15)
    @test rVt*rVt' ≈ Matrix(I, size(Vt, 1), size(Vt, 1))
end
end # @testset "Tensor factorizations"
