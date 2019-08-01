using Test, MPStates, LinearAlgebra

# These are just integration tests to test that the DMRG algorithms don't error.
@testset "Integration tests: variational/DMRG algorithms" begin
@testset "Variational simplification of Mps" begin
    rtest1 = MPStates.init_test_mps("rtest1")
    simplify!(rtest1, 1)
end

@testset "DMRG algorithms" for T in [Float64, ComplexF64]
    # Diagonalize a simple Hamiltonian with first neighbor interactions and test
    # that energy and variance decrease after each sweep and that the method
    # doesn't error.
    L = 10
    J = Hermitian(convert.(T, diagm(1 => 1:L-1)))
    H = init_mpo(T, L, 2)
    add_ops!(H, "b+", "b", convert.(T, J))

    # Max allowed bond dimension.
    m = 20
    # Maximum number of iterations.
    max_iters = 10

    # DMRG1.
    psi = init_mps(T, L, "W")
    E1, var = minimize!(psi, H, m, "DMRG1", debug=0, max_iters=max_iters)
    for i=2:max_iters
        @test E1[i] < E1[i-1]
    end

    # DMRG2.
    psi = init_mps(T, L, "W")
    E2, var = minimize!(psi, H, m, "DMRG2", debug=0, max_iters=max_iters)
    for i=2:max_iters
        @test E2[i] < E2[i-1]
    end

    # DMRG3S.
    psi = init_mps(T, L, "W")
    E3, var = minimize!(psi, H, m, "DMRG3S", debug=0, max_iters=max_iters)
    for i=2:max_iters
        @test E3[i] < E3[i-1]
    end
end
end # @testset "Integration tests: variational/DMRG algorithms"
