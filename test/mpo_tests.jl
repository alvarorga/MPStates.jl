using Test, MPStates, LinearAlgebra

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

@testset "Mpo with local terms" begin
    L = 6
    d = 2

    Op = init_mpo(Float64, L, d)
    weights = [0.2, 0., 0.3, 0., 0.7, 0.]
    Op = add_ops!(Op, "n", weights)

    rtest1 = MPStates.init_test_mps("rtest1")
    rtest2 = MPStates.init_test_mps("rtest2")
    @test expected(Op, rtest1) ≈ 0.2*1/9 + 0.3*4/9 + 0.7*0.64
    @test expected(Op, rtest2) ≈ 0.2*5/9 + 0.3 + 0.7

    cOp = init_mpo(ComplexF64, L, d)
    cweights = ComplexF64[1., 2., 0., 0., 0., 0.3]
    add_ops!(cOp, "n", cweights)

    ctest1 = MPStates.init_test_mps("ctest1")
    ctest2 = MPStates.init_test_mps("ctest2")
    @test expected(cOp, ctest1) ≈ 1/9 + 2*4/9 + 0.3
    @test expected(cOp, ctest2) ≈ 5/9 + 2*5/9 + 0.3*0.64
end

@testset "Mpo with two-body terms" begin
    L = 6
    d = 2

    # Test with operators n_i*n_j.
    Op = init_mpo(Float64, L, d)
    V = (diagm(0 => [0., 0.8, 0., 0.5, 0., 0.])
         .+ diagm(1 => [0.3, 0., 0.2, 0.1, 0.])
         .+ diagm(2 => [0.4, 0.7, 0.6, 0.]))
    V = Symmetric(V)
    Op = add_ops!(Op, "n", "n", V)

    rtest1 = MPStates.init_test_mps("rtest1")
    rtest2 = MPStates.init_test_mps("rtest2")
    @test expected(Op, rtest1) ≈ (0.8*4/9 + 0.5 + 0.4*4/9 + 0.2*0.64 + 1.4*4/9
                                  + 1.2*4/9*0.64)
    @test expected(Op, rtest2) ≈ (0.8*5/9 + 0.5 + 0.6*1/9 + 0.4 + 0.2 + 0.8*5/9
                                  + 1.4*5/9 + 1.2)

    cOp = init_mpo(ComplexF64, L, d)
    cV = convert.(ComplexF64, V)
    add_ops!(cOp, "n", "n", cV)

    ctest1 = MPStates.init_test_mps("ctest1")
    ctest2 = MPStates.init_test_mps("ctest2")
    @test expected(cOp, ctest1) ≈ (0.8*4/9 + 0.5 + 0.4*4/9 + 0.2*0.64 + 1.4*4/9
                                   + 1.2*4/9*0.64)
    @test expected(cOp, ctest2) ≈ (0.8*5/9 + 0.5 + 0.6*1/9 + 0.4 + 0.2 + 0.8*5/9
                                   + 1.4*5/9 + 1.2)
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
