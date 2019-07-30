using Test, MPStates, LinearAlgebra

@testset "Operations with Mpo" begin
# Initialize testing Mps.
rtest1 = MPStates.init_test_mps("rtest1")
ctest1 = MPStates.init_test_mps("ctest1")
rtest2 = MPStates.init_test_mps("rtest2")
ctest2 = MPStates.init_test_mps("ctest2")

@testset "initialize empty Mpo" begin
    L = 6
    d = 2
    Op = init_mpo(Float64, L, d)
    cOp = init_mpo(ComplexF64, L, d)

    @test expected(Op, rtest1) ≈ 0. atol=1e-15
    @test expected(cOp, ctest1) ≈ 0. atol=1e-15
    @test expected(Op, rtest2) ≈ 0. atol=1e-15
    @test expected(cOp, ctest2) ≈ 0. atol=1e-15
end

@testset "Mpo with local terms" begin
    L = 6
    d = 2

    Op = init_mpo(Float64, L, d)
    weights = [0.2, 0., 0.3, 0., 0.7, 0.]
    add_ops!(Op, "n", weights)

    @test expected(Op, rtest1) ≈ 0.2*1/9 + 0.3*4/9 + 0.7*0.64
    @test expected(Op, rtest2) ≈ 0.2*5/9 + 0.3 + 0.7

    cOp = init_mpo(ComplexF64, L, d)
    cweights = complex.([1., 2., 0., 0., 0., 0.3], 0.)
    add_ops!(cOp, "n", cweights)

    @test expected(cOp, ctest1) ≈ 1/9 + 2*4/9 + 0.3
    @test expected(cOp, ctest2) ≈ 5/9 + 2*5/9 + 0.3*0.64
end

@testset "Mpo with n_i*n_j terms" begin
    L = 6
    d = 2

    # Operator matrix.
    V = (diagm(0 => [0., 0.8, 0., 0.5, 0., 0.])
         .+ diagm(1 => [0.3, 0., 0.2, 0.1, 0.])
         .+ diagm(2 => [0.4, 0.7, 0.6, 0.]))
    V = Symmetric(V)

    # Tests with real Mps.
    Op = init_mpo(Float64, L, d)
    add_ops!(Op, "n", "n", V)

    @test expected(Op, rtest1) ≈ (0.8*4/9 + 0.5 + 0.4*4/9 + 0.2*0.64 + 1.4*4/9
                                  + 1.2*4/9*0.64)
    @test expected(Op, rtest2) ≈ (0.8*5/9 + 0.5 + 0.6*1/9 + 0.4 + 0.2 + 0.8*5/9
                                  + 1.4*5/9 + 1.2)

    # Tests with complex Mps.
    cOp = init_mpo(ComplexF64, L, d)
    add_ops!(cOp, "n", "n", convert.(ComplexF64, V))

    @test expected(cOp, ctest1) ≈ (0.8*4/9 + 0.5 + 0.4*4/9 + 0.2*0.64 + 1.4*4/9
                                   + 1.2*4/9*0.64)
    @test expected(cOp, ctest2) ≈ (0.8*5/9 + 0.5 + 0.6*1/9 + 0.4 + 0.2 + 0.8*5/9
                                   + 1.4*5/9 + 1.2)
end

@testset "Mpo with n_i*n_j and fermionic c+_i*c_j terms" begin
    L = 6
    d = 2

    # Operator matrices.
    V = (diagm(0 => [0., 0.8, 0., 0.5, 0., 0.])
         .+ diagm(1 => [0.3, 0., 0.2, 0.1, 0.])
         .+ diagm(2 => [0.4, 0.7, 0.6, 0.]))
    V = Symmetric(V)
    J = zeros(L, L)
    J[1, 2] = 0.5
    J[2, 1] = 0.3
    J[1, 5] = 0.4
    J[5, 1] = 0.3
    J[1, 6] = 0.7
    J[6, 1] = 0.8
    J[2, 2] = 0.1
    J[2, 6] = 0.9
    J[6, 2] = 1.2

    # Test with real Mps.
    Op = init_mpo(Float64, L, d)
    add_ops!(Op, "n", "n", V)
    add_ops!(Op, "c+", "c", J, ferm_op="Z")

    @test expected(Op, rtest1) ≈ (0.8*4/9 + 0.5 + 0.4*4/9 + 0.2*0.64 + 1.4*4/9
                                  + 1.2*4/9*0.64 - 0.5*2/9 - 0.3*2/9 + 0.1*4/9)
    @test expected(Op, rtest2) ≈ (0.8*5/9 + 0.5 + 0.6*1/9 + 0.4 + 0.2 + 0.8*5/9
                                  + 1.4*5/9 + 1.2 - 0.5*4/9 - 0.3*4/9
                                  + 0.7*2/9*0.8*0.6 + 0.8*2/9*0.8*0.6
                                  + 0.1*5/9 + 0.9*2/9*0.8*0.6 + 1.2*2/9*0.8*0.6)

    # Test with complex Mps.
    cOp = init_mpo(ComplexF64, L, d)
    add_ops!(cOp, "n", "n", convert.(ComplexF64, V))
    add_ops!(cOp, "c+", "c", convert.(ComplexF64, J), ferm_op="Z")

    @test expected(cOp, ctest1) ≈ (0.8*4/9 + 0.5 + 0.4*4/9 + 0.2*0.64 + 1.4*4/9
                                   + 1.2*4/9*0.64 + 0.5*complex(0, 2/9)
                                   + 0.3*complex(0, -2/9) + 0.1*4/9)
    @test expected(cOp, ctest2) ≈ (0.8*5/9 + 0.5 + 0.6*1/9 + 0.4 + 0.2 + 0.8*5/9
                                   + 1.4*5/9 + 1.2 + 0.5*complex(0, -4/9)
                                   + 0.3*complex(0, 4/9)
                                   + 0.7*2/9*0.8*0.6*complex(0., 1.)
                                   + 0.8*2/9*0.8*0.6*complex(0., -1.)
                                   + 0.1*5/9 + 0.9*2/9*0.8*0.6 + 1.2*2/9*0.8*0.6)
end

@testset "Mpo with non-fermionic b+_i*b_j terms" begin
    L = 6
    d = 2

    # Operator matrix.
    J = zeros(L, L)
    J[1, 2] = 0.5
    J[2, 1] = 0.3
    J[1, 5] = 0.4
    J[5, 1] = 0.3
    J[1, 6] = 0.7
    J[6, 1] = 0.8
    J[2, 2] = 0.1
    J[2, 6] = 0.9
    J[6, 2] = 1.2

    # Test with real Mps.
    Op = init_mpo(Float64, L, d)
    add_ops!(Op, "c+", "c", J)

    @test expected(Op, rtest1) ≈ - 0.5*2/9 - 0.3*2/9 + 0.1*4/9
    @test expected(Op, rtest2) ≈ (- 0.5*4/9 - 0.3*4/9 + 0.7*2/9*0.8*0.6
                                  + 0.8*2/9*0.8*0.6 + 0.1*5/9 - 0.9*2/9*0.8*0.6
                                  - 1.2*2/9*0.8*0.6)

    # Test with real complex Mps.
    cOp = init_mpo(ComplexF64, L, d)
    add_ops!(cOp, "c+", "c", convert.(ComplexF64, J))

    @test expected(cOp, ctest1) ≈ (0.5*complex(0, 2/9) + 0.3*complex(0, -2/9)
                                   + 0.1*4/9)
    @test expected(cOp, ctest2) ≈ (0.5*complex(0, -4/9) + 0.3*complex(0, 4/9)
                                   + 0.7*2/9*0.8*0.6*complex(0., 1.)
                                   + 0.8*2/9*0.8*0.6*complex(0., -1.)
                                   + 0.1*5/9 - 0.9*2/9*0.8*0.6 - 1.2*2/9*0.8*0.6)

    # Test with the "GHZ", "full", "W", and "product" states.
    J = zeros(L, L)
    for i=1:L-1
        J[i, i+1] = i/10
        J[i+1, i] = 3. *i
    end
    J[1, L] = 0.5
    J[L, 1] = -2.

    V = zeros(L, L)
    for i=1:L-1
        V[i, i+1] = i/4
        V[i+1, i] = -1.5*i
    end
    V[1, L] = 0.7
    V[L, 1] = -4.

    Op = init_mpo(Float64, L, 2)
    add_ops!(Op, "b+", "b", J)
    add_ops!(Op, "n", "n", V)

    GHZ = init_mps(Float64, L, "GHZ")
    W = init_mps(Float64, L, "W")
    full = init_mps(Float64, L, "full")
    product = init_mps(Float64, L, "product")
    @test expected(Op, GHZ) ≈ (15/4 - 1.5*15 + 0.7 - 4.)/2.
    @test expected(Op, W) ≈ (15/10 + 3*15 + 0.5 -2.)/L
    @test expected(Op, full) ≈ (15/10 + 3*15 + 0.5 -2. + 15/4 - 15*1.5 + 0.7
                                - 4.)/4
    @test expected(Op, product) ≈ 0.
end

@testset "Mpo with fermionic c+_i*c_j terms" begin
    L = 4
    J = zeros(L, L)
    for i=1:L-2
        J[i, i+2] = i/5
        J[i+2, i] = 2. *i
    end
    J[1, L] = 0.7
    J[L, 1] = -2.3

    Op = init_mpo(Float64, L, 2)
    add_ops!(Op, "c+", "c", J, ferm_op="Z")

    full = init_mps(Float64, L, "full")
    @test expected(Op, full) ≈ 0. atol=1e-15
end
end # @testset "Operations with Mpo"
