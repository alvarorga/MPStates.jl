using Test, MPStates

@testset "Operations with Mpo" begin
@testset "initialize empty Mpo" begin
    L = 6
    d = 2
    Op = init_mpo(Float64, L, d)
    cOp = init_mpo(ComplexF64, L, d)

    rtest1 = MPStates.init_test_mps("rtest1")
    @test expected(Op, rtest1) ≈ 0. atol=1e-15
    ctest1 = MPStates.init_test_mps("ctest1")
    @test expected(cOp, ctest1) ≈ 0. atol=1e-15
    rtest2 = MPStates.init_test_mps("rtest2")
    @test expected(Op, rtest2) ≈ 0. atol=1e-15
    ctest2 = MPStates.init_test_mps("ctest2")
    @test expected(cOp, ctest2) ≈ 0. atol=1e-15
end

@testset "make Hubbard MPO" begin
    L = 5
    t = 1.
    U = 0.5

    H = init_hubbard_mpo(L, t, U)
end

@testset "build general MPOs" begin
    L = 6
    J = zeros(L, L)
    V = zeros(L, L)

    for i=1:L-1
        J[i, i+1] = i/10
        J[i+1, i] = 3. *i
    end
    J[1, L] = 0.5
    J[L, 1] = -2.

    for i=1:L-1
        V[i, i+1] = i/4
        V[i+1, i] = -1.5*i
    end
    V[1, L] = 0.7
    V[L, 1] = -4.

    Op = init_mpo(L, J, V, false)

    GHZ = init_mps(Float64, L, "GHZ")
    W = init_mps(Float64, L, "W")
    full = init_mps(Float64, L, "full")
    product = init_mps(Float64, L, "product")

    @test expected(Op, GHZ) ≈ (15/4 - 1.5*15 + 0.7 - 4.)/2.
    @test expected(Op, W) ≈ (15/10 + 3*15 + 0.5 -2.)/L
    @test expected(Op, full) ≈ (15/10 + 3*15 + 0.5 -2. + 15/4 - 15*1.5 + 0.7 - 4.)/4
    @test expected(Op, product) ≈ 0.

    # Test a fermionic Mpo.
    L = 4
    J = zeros(L, L)
    V = zeros(L, L)

    for i=1:L-2
        J[i, i+2] = i/5
        J[i+2, i] = 2. *i
    end
    J[1, L] = 0.7
    J[L, 1] = -2.3

    Op = init_mpo(L, J, V, true)
    full = init_mps(Float64, L, "full")
    @test expected(Op, full) ≈ 0. atol=1e-15
end
end # @testset "Operations with Mpo"
