using MPStates, Test

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

@testset "apply Mpo to Mps" begin
    L = 6
    t = 1.5
    U = 1.5
    J = zeros(L, L)
    J[1, 2] = 1.
    J[4, 5] = 1.
    V = zeros(L, L)
    Op = init_mpo(L, J, V, true)
    rtest1 = MPStates.init_test_mps("rtest1")
    apply!(Op, rtest1)
    MPStates.make_left_canonical!(rtest1, false)
    @test norm(rtest1) ≈ 4/9
    @test m_occupation(rtest1, 1) ≈ 4/9
    @test m_occupation(rtest1, 2) ≈ 0. atol=1e-15
    @test m_occupation(rtest1, 3) ≈ 0. atol=1e-15
    @test m_occupation(rtest1, 4) ≈ 4/9
    @test m_occupation(rtest1, 5) ≈ 4/9*0.64
    @test m_occupation(rtest1, 6) ≈ 4/9
    ctest2 = MPStates.init_test_mps("ctest2")
    Op = init_mpo(L, complex.(J), complex.(V), true)
    apply!(Op, ctest2)
    MPStates.make_left_canonical!(ctest2, false)
    @test norm(ctest2) ≈ 4/9
    @test m_occupation(ctest2, 1) ≈ 4/9
    @test m_occupation(ctest2, 2) ≈ 0. atol=1e-15
    @test m_occupation(ctest2, 3) ≈ 4/9
    @test m_occupation(ctest2, 4) ≈ 4/9
    @test m_occupation(ctest2, 5) ≈ 4/9
    @test m_occupation(ctest2, 6) ≈ 4/9*0.64
end
end # @testset "Operations with Mpo and Mps"
