@testset "Operations with Mpo and Mps" begin
@testset "expectation value of Mps" begin
    L = 5
    t = 1.5
    U = 1.5
    H = init_hubbard_mpo(L, t, U)
    product = init_mps(Float64, L, "product")
    @test expected(H, product) ≈ 0. atol=1e-15
    GHZ = init_mps(Float64, L, "GHZ")
    @test expected(H, GHZ) ≈ U/2*(L-1)
    W = init_mps(Float64, L, "W")
    @test expected(H, W) ≈ 2t*(L-1)/L
end

@testset "expectation value of squared Mps" begin
    L = 5
    t = 1.5
    U = 1.5
    H = init_hubbard_mpo(L, t, U)
    product = init_mps(Float64, L, "product")
    @test MPStates.expected2(H, product) ≈ 0. atol=1e-15
    GHZ = init_mps(Float64, L, "GHZ")
    @test MPStates.expected2(H, GHZ) ≈ U^2/2*(L-1)^2
    W = init_mps(Float64, L, "W")
    @test MPStates.expected2(H, W) ≈ t^2*(4L-6)/L
end

@testset "variance of Mpo" begin
    L = 5
    t = 1.5
    U = 1.5
    H = init_hubbard_mpo(L, t, U)
    product = init_mps(Float64, L, "product")
    @test m_variance(H, product) ≈ 0. atol=1e-15
    GHZ = init_mps(Float64, L, "GHZ")
    @test m_variance(H, GHZ) ≈ U^2/4*(L-1)^2
    W = init_mps(Float64, L, "W")
    @test m_variance(H, W) ≈ t^2*(4L-6)/L - (2t*(L-1)/L)^2
end
end # @testset "Operations with Mpo and Mps"
