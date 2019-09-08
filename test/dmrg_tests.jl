using Test, Random, MPStates, LinearAlgebra

# These are just integration tests to test that the DMRG algorithms don't error.
@testset "Integration tests: variational/DMRG algorithms" begin
@testset "Variational simplification of Mps" begin
    rtest1 = MPStates.testMps("rtest1")
    simplify!(rtest1, 1)
end

@testset "DMRG algorithms" for T in [Float64, ComplexF64]
    Random.seed!(1)

    # Diagonalize a simple Hamiltonian with first neighbor interactions and test
    # that energy and variance decrease after each sweep and that the method
    # doesn't error.
    L = 15
    J = Hermitian(convert.(T, randn(L, L)))
    d = 2
    H = Mpo(T, L, d)
    add_ops!(H, "b+", "b", convert.(T, J))

    # Max allowed bond dimension per sweep.
    maxm = [10, 20, 20]

    # DMRG1.
    psi = randomMps(T, L, d, 2)
    dmrg_opts = DMRGOpts("DMRG1", maxm, 1e-10, show_trace=0)
    E1, var1 = dmrg!(psi, H, dmrg_opts)

    # DMRG2.
    psi = randomMps(T, L, d, 2)
    dmrg_opts = DMRGOpts("DMRG2", maxm, 1e-10, show_trace=0)
    E2, var2 = dmrg!(psi, H, dmrg_opts)

    # DMRG3S.
    psi = randomMps(T, L, d, 2)
    dmrg_opts = DMRGOpts("DMRG3S", maxm, 1e-10, show_trace=0)
    E3, var3 = dmrg!(psi, H, dmrg_opts)
end
end # @testset "Integration tests: variational/DMRG algorithms"
