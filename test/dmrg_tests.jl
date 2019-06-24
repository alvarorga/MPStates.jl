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
    D = zeros(T, L, L)
    is_fermionic = true
    H = MPStates.init_mpo(L, J, D, is_fermionic)
    # Max allowed bond dimension.
    max_D = 10

    # DMRG1.
    psi = MPStates.init_mps(T, L, "W")
    E, var = MPStates.minimize!(psi, H, max_D, "DMRG1", debug=0)

    # DMRG2.
    psi = MPStates.init_mps(T, L, "W")
    E, var = MPStates.minimize!(psi, H, max_D, "DMRG2", debug=0)
end
end # @testset "Integration tests: variational/DMRG algorithms"
