using MPStates, Test

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
    @test m_fermionic_correlation(full, 3, 4) ≈ 0.25
    @test m_fermionic_correlation(full, 3, 2) ≈ 0.25
    @test m_fermionic_correlation(full, 4, 3) ≈ 0.25
    @test m_fermionic_correlation(full, 1, 4) ≈ 0. atol=1e-15
    @test m_fermionic_correlation(full, 4, 2) ≈ 0. atol=1e-15
    @test m_fermionic_correlation(full, 1, 3) ≈ 0. atol=1e-15
    @test m_fermionic_correlation(full, 2, 4) ≈ 0. atol=1e-15
end

@testset "measure correlations" begin
    L = 4
    full = init_mps(Float64, L, "full")
    @test m_correlation(full, 1, 2) ≈ 0.25
    @test m_correlation(full, 2, 3) ≈ 0.25
    @test m_correlation(full, 2, 4) ≈ 0.25
    @test m_correlation(full, 1, 4) ≈ 0.25
    @test m_correlation(full, 3, 2) ≈ 0.25
    @test m_correlation(full, 4, 2) ≈ 0.25
end

@testset "measure 2 point occupations" begin
    L = 5
    GHZ = init_mps(Float64, L, "GHZ")
    @test m_2occupations(GHZ, 1, 3) ≈ 0.5
    @test m_2occupations(GHZ, 3, 2) ≈ 0.5
    @test m_2occupations(GHZ, 4, 2) ≈ 0.5
    @test m_2occupations(GHZ, 1, 5) ≈ 0.5
    full = init_mps(Float64, L, "full")
    @test m_2occupations(full, 1, 3) ≈ 0.25
    @test m_2occupations(full, 3, 2) ≈ 0.25
    @test m_2occupations(full, 4, 3) ≈ 0.25
    @test m_2occupations(full, 1, 5) ≈ 0.25
end

@testset "contraction of two MPS" begin
    L = 5
    GHZ = init_mps(Float64, L, "GHZ")
    W = init_mps(Float64, L, "W")
    full = init_mps(Float64, L, "full")
    product = init_mps(Float64, L, "product")
    @test contract(GHZ, W) ≈ 0. atol=1e-15
    @test contract(GHZ, full) ≈ 1/sqrt(2^(L-1))
    @test contract(GHZ, product) ≈ 1/sqrt(2)
    @test contract(W, full) ≈ sqrt(L/2^L)
    @test contract(W, product) ≈ 0. atol=1e-15
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

@testset "enlargement of bond dimension of MPS" begin
    L = 10
    GHZ = init_mps(Float64, L, "GHZ")
    enlarge_bond_dimension!(GHZ, 5)
    @test size(GHZ.A[1], 1) == 1
    @test size(GHZ.A[1], 3) == 2
    @test size(GHZ.A[3], 1) == 4
    @test size(GHZ.A[3], 3) == 5
    for i=4:7
        @test size(GHZ.A[i], 1) == 5
        @test size(GHZ.A[i], 3) == 5
    end
    @test size(GHZ.A[8], 1) == 5
    @test size(GHZ.A[8], 3) == 4
    @test size(GHZ.A[L], 1) == 2
    @test size(GHZ.A[L], 3) == 1
    # Check that the properties of the Mps are left intact.
    full = init_mps(Float64, L, "full")
    @test contract(GHZ, full) ≈ 1/sqrt(2^(L-1))
    enlarge_bond_dimension!(full, 11)
    @test contract(GHZ, full) ≈ 1/sqrt(2^(L-1))
end

@testset "SVD truncation of MPS" begin
    L = 10
    GHZ = init_mps(Float64, L, "GHZ")
    enlarge_bond_dimension!(GHZ, 5)
    svd_truncate!(GHZ, 3)
    @test size(GHZ.A[1], 1) == 1
    @test size(GHZ.A[1], 3) == 2
    @test size(GHZ.A[2], 1) == 2
    @test size(GHZ.A[2], 3) == 3
    for i=3:8
        @test size(GHZ.A[i], 1) == 3
        @test size(GHZ.A[i], 3) == 3
    end
    @test size(GHZ.A[9], 1) == 3
    @test size(GHZ.A[9], 3) == 2
    @test size(GHZ.A[10], 1) == 2
    @test size(GHZ.A[10], 3) == 1
    # Check that the properties of the Mps are left intact.
    full = init_mps(Float64, L, "full")
    @test contract(GHZ, full) ≈ 1/sqrt(2^(L-1))
end

@testset "save and read Mps in hdf5 format" for T in [Float64, ComplexF64]
    L = 10
    GHZ = init_mps(T, L, "GHZ")

    filename = "foo.h5"
    # Remove file if previous test crashed and file was not removed.
    isfile(filename) && rm(filename)

    # Save Mps.
    @test save_mps(filename, GHZ, true) == 0
    # Read Mps.
    psi = read_mps(filename)
    @test psi.L == L
    @test psi.d == 2
    @test eltype(psi.A[1]) == T
    for i=1:L
        @test psi.A[i] ≈ GHZ.A[i]
        @test psi.B[i] ≈ GHZ.B[i]
    end
    # Remove hdf5 testing file.
    isfile(filename) && rm(filename)

    # Make test again with save_B=false.
    @test save_mps(filename, GHZ, false) == 0
    # Read Mps.
    psi = read_mps(filename)
    @test psi.L == L
    @test psi.d == 2
    @test eltype(psi.A[1]) == T
    for i=1:L
        @test psi.A[i] ≈ GHZ.A[i]
        # Test with abs() because some signs might be changed/gauged.
        @test abs.(psi.B[i]) ≈ abs.(GHZ.B[i])
    end
    # Remove hdf5 testing file.
    isfile(filename) && rm(filename)
end
