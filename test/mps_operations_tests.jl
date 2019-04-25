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

@testset "measure fermionic correlations" begin
    L = 4
    full = init_mps(Float64, L, "full")
    @test m_fermionic_correlation(full, 1, 2) ≈ 0.25
    @test m_fermionic_correlation(full, 1, 3) ≈ 0.
    @test m_fermionic_correlation(full, 1, 4) ≈ 0.
end

@testset "measure correlations" begin
    L = 4
    full = init_mps(Float64, L, "full")
    @test m_correlation(full, 1, 2) ≈ 0.25
    @test m_correlation(full, 1, 3) ≈ 0.25
    @test m_correlation(full, 1, 4) ≈ 0.25
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

@testset "schmidt decomposition" begin
    L = 4
    GHZ = init_mps(Float64, L, "GHZ")
    @test MPStates.schmidt_decomp(GHZ, 1) ≈ [1/sqrt(2), 1/sqrt(2)]
    @test MPStates.schmidt_decomp(GHZ, 2) ≈ [1/sqrt(2), 1/sqrt(2)]
    W = init_mps(Float64, L, "W")
    @test MPStates.schmidt_decomp(W, 1) ≈ [sqrt(3)/2, 0.5]
    @test MPStates.schmidt_decomp(W, 2) ≈ [1/sqrt(2), 1/sqrt(2)]
end

@testset "entanglement entropy" begin
    L = 4
    GHZ = init_mps(Float64, L, "GHZ")
    @test ent_entropy(GHZ, 1) ≈ 1/sqrt(2)
    @test ent_entropy(GHZ, 2) ≈ 1/sqrt(2)
    W = init_mps(Float64, L, "W")
    @test ent_entropy(W, 1) ≈ 0.5 - sqrt(3)/2*log2(sqrt(3)/2)
    @test ent_entropy(W, 2) ≈ 1/sqrt(2)
end
