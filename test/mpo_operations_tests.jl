using MPStates, Test, LinearAlgebra

@testset "Operations with Mpo and Mps" begin
# Define a Hubbard Hamiltonian that we will use for testing.
L = 6
# Physical dimension.
d = 2
t = 1.5
U = 1.5
J = t .* Symmetric(diagm(1 => ones(L-1)))
V = U .* diagm(1 => ones(L-1))

H = init_mpo(Float64, L, d)
# Add fermionic operators.
add_ops!(H, "c+", "c", J, ferm_op="Z")
# Add interaction terms.
add_ops!(H, "n", "n", V)

# Define "W", "GHZ", and "product" states.
W = init_mps(Float64, L, "W")
GHZ = init_mps(Float64, L, "GHZ")
product = init_mps(Float64, L, "product")

# TODO: add testing Mps.

@testset "expectation value of Mps" begin
    @test expected(H, product) ≈ 0. atol=1e-15
    @test expected(H, GHZ) ≈ U/2*(L-1)
    @test expected(H, W) ≈ 2t*(L-1)/L
end

@testset "expectation value of squared Mps" begin
    @test MPStates.norm_of_apply(H, product) ≈ 0. atol=1e-15
    @test MPStates.norm_of_apply(H, GHZ) ≈ U^2/2*(L-1)^2
    @test MPStates.norm_of_apply(H, W) ≈ t^2*(4L-6)/L
end

@testset "variance of Mpo" begin
    @test m_variance(H, product) ≈ 0. atol=1e-15
    @test m_variance(H, GHZ) ≈ U^2/4*(L-1)^2
    @test m_variance(H, W) ≈ t^2*(4L-6)/L - (2t*(L-1)/L)^2
end

@testset "apply Mpo to Mps" begin
    L = 6
    J = zeros(L, L)
    J[1, 2] = 1.
    J[4, 5] = 1.

    rtest1 = MPStates.init_test_mps("rtest1")
    ctest2 = MPStates.init_test_mps("ctest2")

    Op = init_mpo(Float64, L, 2)
    add_ops!(Op, "c+", "c", J, ferm_op="Z")

    apply!(Op, rtest1)
    MPStates.make_left_canonical!(rtest1, false)
    @test norm(rtest1) ≈ 4/9
    @test expected(rtest1, "n", 1) ≈ 4/9
    @test expected(rtest1, "n", 2) ≈ 0. atol=1e-15
    @test expected(rtest1, "n", 3) ≈ 0. atol=1e-15
    @test expected(rtest1, "n", 4) ≈ 4/9
    @test expected(rtest1, "n", 5) ≈ 4/9*0.64
    @test expected(rtest1, "n", 6) ≈ 4/9

    cOp = init_mpo(ComplexF64, L, 2)
    add_ops!(cOp, "c+", "c", convert.(ComplexF64, J), ferm_op="Z")

    apply!(cOp, ctest2)
    MPStates.make_left_canonical!(ctest2, false)
    @test norm(ctest2) ≈ 4/9
    @test expected(ctest2, "n", 1) ≈ 4/9
    @test expected(ctest2, "n", 2) ≈ 0. atol=1e-15
    @test expected(ctest2, "n", 3) ≈ 4/9
    @test expected(ctest2, "n", 4) ≈ 4/9
    @test expected(ctest2, "n", 5) ≈ 4/9
    @test expected(ctest2, "n", 6) ≈ 4/9*0.64
end
end # @testset "Operations with Mpo and Mps"
