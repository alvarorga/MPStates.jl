@testset "expectation value of Mps" begin
    L = 5
    t = 1.5
    U = 1.5
    H = init_hubbard_mpo(L, t, U)
    product = init_mps(Float64, L, "product")
    @test expected(H, product) ≈ 0.
    GHZ = init_mps(Float64, L, "GHZ")
    @test expected(H, GHZ) ≈ U/2*(L-1)
    W = init_mps(Float64, L, "W")
    @test expected(H, W) ≈ 2t*(L-1)/L
end
