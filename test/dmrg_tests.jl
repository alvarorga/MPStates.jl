using Test, MPStates

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
    J = zeros(T, L, L)
    for i=1:L-1
        J[i,i+1] = i
        J[i+1, i] = i
    end
    H = init_mpo(T, L, 2)
    add_ops!(H, "b+", "b", convert.(T, J))
    # Max allowed bond dimension.
    m = 10

    # DMRG1.
    psi = init_mps(T, L, "W")
    E, var = minimize!(psi, H, m, "DMRG1", debug=0)

    # DMRG2.
    psi = init_mps(T, L, "W")
    E, var = minimize!(psi, H, m, "DMRG2", debug=0)

    # DMRG3S.
    psi = init_mps(T, L, "W")
    E, var = minimize!(psi, H, m, "DMRG3S", debug=0)
end
end # @testset "Integration tests: variational/DMRG algorithms"
