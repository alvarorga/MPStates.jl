using Test, MPStates, LinearAlgebra

# These are just integration tests to test that the DMRG algorithms don't error.
@testset "Integration tests: variational/DMRG algorithms" begin
@testset "Variational simplification of Mps" begin
    rtest1 = MPStates.testMps("rtest1")
    simplify!(rtest1, 1)
end

@testset "DMRG algorithms" for T in [Float64, ComplexF64]
    # Diagonalize a simple Hamiltonian with first neighbor interactions and test
    # that energy and variance decrease after each sweep and that the method
    # doesn't error.
    L = 10
    J = Hermitian(convert.(T, diagm(1 => 1:L-1)))
    d = 2
    H = Mpo(T, L, d)
    add_ops!(H, "b+", "b", convert.(T, J))

    # Max allowed bond dimension.
    m = 20

    # DMRG1.
    psi = randomMps(T, L, d, 2)
    min_opts = MinimizeOpts(m, "DMRG1", debug=0)
    E1, var1 = minimize!(psi, H, min_opts)

    # DMRG2.
    psi = randomMps(T, L, d, 2)
    min_opts = MinimizeOpts(m, "DMRG2", debug=0)
    E2, var2 = minimize!(psi, H, min_opts)

    # DMRG3S.
    psi = randomMps(T, L, d, 2)
    min_opts = MinimizeOpts(m, "DMRG3S", debug=0)
    E3, var3 = minimize!(psi, H, min_opts)
end
end # @testset "Integration tests: variational/DMRG algorithms"
