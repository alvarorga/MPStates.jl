using Test, MPStates, LinearAlgebra, Random

@testset "Operations with Mpo" begin
# Initialize testing Mps.
rtest1 = MPStates.testMps("rtest1")
ctest1 = MPStates.testMps("ctest1")
rtest2 = MPStates.testMps("rtest2")
ctest2 = MPStates.testMps("ctest2")
rtest3 = MPStates.testMps("rtest3")

@testset "initialize empty Mpo" begin
    L = 6
    d = 2
    Op = Mpo(L, d)
    cOp = Mpo(ComplexF64, L, d)
    sOp = Mpo(L, 3)

    @test expected(Op, rtest1) ≈ 0. atol=1e-15
    @test expected(Op, rtest2) ≈ 0. atol=1e-15
    @test expected(sOp, rtest3) ≈ 0. atol=1e-15
    @test expected(cOp, ctest1) ≈ 0. atol=1e-15
    @test expected(cOp, ctest2) ≈ 0. atol=1e-15
end

@testset "Mpo with local terms" begin
    L = 6
    d = 2

    Op = Mpo(L, d)
    weights = cos.(1:L)
    add_ops!(Op, "n", weights)

    res1 = 0.
    res2 = 0.
    for i=1:L
        res1 += expected(rtest1, "n", i)*weights[i]
        res2 += expected(rtest2, "n", i)*weights[i]
    end
    @test expected(Op, rtest1) ≈ res1
    @test expected(Op, rtest2) ≈ res2

    cOp = Mpo(ComplexF64, L, d)
    cweights = complex.(tan.(1:L), sin.(1:L))
    add_ops!(cOp, "n", cweights)

    cres1 = complex(0.)
    cres2 = complex(0.)
    for i=1:L
        cres1 += expected(ctest1, "n", i)*cweights[i]
        cres2 += expected(ctest2, "n", i)*cweights[i]
    end
    @test expected(cOp, ctest1) ≈ cres1
    @test expected(cOp, ctest2) ≈ cres2

    # Test with spin states.
    d = 3

    Op = Mpo(L, d)
    weights1 = cos.(1:L)
    weights2 = tan.(1:L)
    add_ops!(Op, "Sz", weights1)
    add_ops!(Op, "S+", weights2)

    res3 = 0.
    for i=1:L
        res3 += expected(rtest3, "Sz", i)*weights1[i]
        res3 += expected(rtest3, "S+", i)*weights2[i]
    end
    @test expected(Op, rtest3) ≈ res3
end

@testset "Mpo with n_i*n_j terms" begin
    L = 6
    d = 2

    # Operator matrix.
    V = reshape(sin.(1:L^2), L, L)

    # Tests with real Mps.
    Op = Mpo(L, d)
    add_ops!(Op, "n", "n", V)

    res1 = 0.
    res2 = 0.
    for i=1:L, j=1:L
        res1 += expected(rtest1, "n", i, "n", j)*V[i, j]
        res2 += expected(rtest2, "n", i, "n", j)*V[i, j]
    end
    @test expected(Op, rtest1) ≈ res1
    @test expected(Op, rtest2) ≈ res2

    # Tests with complex Mps.
    cOp = Mpo(ComplexF64, L, d)
    cV = complex.(reshape(sin.(1:L^2), L, L), reshape(cos.(1:L^2), L, L))
    add_ops!(cOp, "n", "n", cV)

    cres1 = 0.
    cres2 = 0.
    for i=1:L, j=1:L
        cres1 += expected(ctest1, "n", i, "n", j)*cV[i, j]
        cres2 += expected(ctest2, "n", i, "n", j)*cV[i, j]
    end
    @test expected(cOp, ctest1) ≈ cres1
    @test expected(cOp, ctest2) ≈ cres2
end

@testset "Mpo with n_i*n_j and fermionic c+_i*c_j terms" begin
    L = 6
    d = 2

    # Operator matrices.
    V = reshape(sin.(1:L^2).^2, L, L)
    J = reshape(tan.(1:L^2).^3, L, L)

    # Test with real Mps.
    Op = Mpo(L, d)
    add_ops!(Op, "n", "n", V)
    add_ops!(Op, "c+", "c", J, ferm_op="Z")

    res1 = 0.
    res2 = 0.
    for i=1:L, j=1:L
        res1 += expected(rtest1, "n", i, "n", j)*V[i, j]
        res1 += expected(rtest1, "c+", i, "c", j, ferm_op="Z")*J[i, j]
        res2 += expected(rtest2, "n", i, "n", j)*V[i, j]
        res2 += expected(rtest2, "c+", i, "c", j, ferm_op="Z")*J[i, j]
    end
    @test expected(Op, rtest1) ≈ res1
    @test expected(Op, rtest2) ≈ res2

    # Tests with complex Mps.
    cOp = Mpo(ComplexF64, L, d)
    cV = complex.(reshape(sin.(1:L^2).^2, L, L), reshape(sin.(1:L^2), L, L))
    cJ = complex.(reshape(tan.(1:L^2).^3, L, L), reshape(tan.(1:L^2), L, L))
    add_ops!(cOp, "n", "n", cV)
    add_ops!(cOp, "c+", "c", cJ, ferm_op="Z")

    cres1 = 0.
    cres2 = 0.
    for i=1:L, j=1:L
        cres1 += expected(ctest1, "n", i, "n", j)*cV[i, j]
        cres1 += expected(ctest1, "c+", i, "c", j, ferm_op="Z")*cJ[i, j]
        cres2 += expected(ctest2, "n", i, "n", j)*cV[i, j]
        cres2 += expected(ctest2, "c+", i, "c", j, ferm_op="Z")*cJ[i, j]
    end
    @test expected(cOp, ctest1) ≈ cres1
    @test expected(cOp, ctest2) ≈ cres2
end

@testset "Mpo with non-fermionic b+_i*b_j terms" begin
    L = 6
    d = 2

    # Operator matrix.
    J = reshape(tan.(1:L^2), L, L)

    # Test with real Mps.
    Op = Mpo(L, d)
    add_ops!(Op, "b+", "b", J)

    res1 = 0.
    res2 = 0.
    for i=1:L, j=1:L
        res1 += expected(rtest1, "b+", i, "b", j)*J[i, j]
        res2 += expected(rtest2, "b+", i, "b", j)*J[i, j]
    end
    @test expected(Op, rtest1) ≈ res1
    @test expected(Op, rtest2) ≈ res2

    # Tests with complex Mps.
    cOp = Mpo(ComplexF64, L, d)
    cJ = complex.(reshape(sin.(1:L^2).^2, L, L), reshape(sin.(1:L^2), L, L))
    add_ops!(cOp, "b+", "b", cJ)

    cres1 = 0.
    cres2 = 0.
    for i=1:L, j=1:L
        cres1 += expected(ctest1, "b+", i, "b", j)*cJ[i, j]
        cres2 += expected(ctest2, "b+", i, "b", j)*cJ[i, j]
    end
    @test expected(cOp, ctest1) ≈ cres1
    @test expected(cOp, ctest2) ≈ cres2
end

@testset "Mpo with fermionic c+_i*c_j terms" begin
    L = 6
    d = 2

    # Operator matrices.
    J = reshape(tan.(1:L^2).*cos.(1:L^2), L, L)

    # Test with real Mps.
    Op = Mpo(L, d)
    add_ops!(Op, "c+", "c", J, ferm_op="Z")

    res1 = 0.
    res2 = 0.
    for i=1:L, j=1:L
        res1 += expected(rtest1, "c+", i, "c", j, ferm_op="Z")*J[i, j]
        res2 += expected(rtest2, "c+", i, "c", j, ferm_op="Z")*J[i, j]
    end
    @test expected(Op, rtest1) ≈ res1
    @test expected(Op, rtest2) ≈ res2

    # Tests with complex Mps.
    cOp = Mpo(ComplexF64, L, d)
    cJ = complex.(reshape(tan.(1:L^2).^2, L, L), reshape(sin.(1:L^2), L, L))
    add_ops!(cOp, "c+", "c", cJ, ferm_op="Z")

    cres1 = 0.
    cres2 = 0.
    for i=1:L, j=1:L
        cres1 += expected(ctest1, "c+", i, "c", j, ferm_op="Z")*cJ[i, j]
        cres2 += expected(ctest2, "c+", i, "c", j, ferm_op="Z")*cJ[i, j]
    end
    @test expected(cOp, ctest1) ≈ cres1
    @test expected(cOp, ctest2) ≈ cres2
end

@testset "Mpo with Sz_i*Sz_j and S+_i*S-_j terms" begin
    L = 6
    d = 3

    # Operator matrices.
    Vzz = reshape(sin.(1:L^2).^2, L, L)
    Vpm = reshape(tan.(1:L^2).^3, L, L)

    # Test with real Mps.
    Op = Mpo(L, d)
    add_ops!(Op, "Sz", "Sz", Vzz)
    add_ops!(Op, "S+", "S-", Vpm)

    res3 = 0.
    for i=1:L, j=1:L
        res3 += expected(rtest3, "Sz", i, "Sz", j)*Vzz[i, j]
        res3 += expected(rtest3, "S+", i, "S-", j)*Vpm[i, j]
    end
    @test expected(Op, rtest3) ≈ res3
end

@testset "expected of hermitian Mpo returns real number" begin
    L = 6
    d = 2

    # Let's create some random complex states to make sure that our premade
    # states have nothing in particular that could oversee some bug.
    Random.seed!(0)
    rtest4 = randomMps(L)
    rtest5 = randomMps(L)
    ctest4 = randomMps(ComplexF64, L)
    ctest5 = randomMps(ComplexF64, L)

    # Operator matrices.
    J = Symmetric(reshape(tan.(1:L^2).*cos.(1:L^2), L, L))
    cJ = Hermitian(complex.(reshape(tan.(1:L^2).^2, L, L),
                            reshape(sin.(1:L^2), L, L)))

    # Test with real symmetric Mpo (or complex with zero imaginary part).
    Op = Mpo(L, d)
    add_ops!(Op, "c+", "c", J, ferm_op="Z")
    @test imag(expected(Op, rtest1)) ≈ 0. atol=1e-8
    @test imag(expected(Op, rtest2)) ≈ 0. atol=1e-8
    @test imag(expected(Op, rtest4)) ≈ 0. atol=1e-8
    @test imag(expected(Op, rtest5)) ≈ 0. atol=1e-8

    Op = Mpo(L, d)
    add_ops!(Op, "c+", "c", J.^3)
    @test imag(expected(Op, rtest1)) ≈ 0. atol=1e-8
    @test imag(expected(Op, rtest2)) ≈ 0. atol=1e-8
    @test imag(expected(Op, rtest4)) ≈ 0. atol=1e-8
    @test imag(expected(Op, rtest5)) ≈ 0. atol=1e-8

    Op = Mpo(L, d)
    add_ops!(Op, "n", "n", J)
    @test imag(expected(Op, rtest1)) ≈ 0. atol=1e-8
    @test imag(expected(Op, rtest2)) ≈ 0. atol=1e-8
    @test imag(expected(Op, rtest4)) ≈ 0. atol=1e-8
    @test imag(expected(Op, rtest5)) ≈ 0. atol=1e-8

    cOp = Mpo(ComplexF64, L, d)
    add_ops!(cOp, "c+", "c", convert.(ComplexF64, J), ferm_op="Z")
    @test imag(expected(cOp, ctest1)) ≈ 0. atol=1e-8
    @test imag(expected(cOp, ctest2)) ≈ 0. atol=1e-8
    # Reduce error tolerance here because errors grow a lot with random states.
    @test imag(expected(cOp, ctest4)) ≈ 0. atol=1e-8
    @test imag(expected(cOp, ctest5)) ≈ 0. atol=1e-8

    cOp = Mpo(ComplexF64, L, d)
    add_ops!(cOp, "c+", "c", convert.(ComplexF64, tan.(J)))
    @test imag(expected(cOp, ctest1)) ≈ 0. atol=1e-8
    @test imag(expected(cOp, ctest2)) ≈ 0. atol=1e-8
    @test imag(expected(cOp, ctest4)) ≈ 0. atol=1e-8
    @test imag(expected(cOp, ctest5)) ≈ 0. atol=1e-8

    cOp = Mpo(ComplexF64, L, d)
    add_ops!(cOp, "n", "n", convert.(ComplexF64, tan.(J)))
    @test imag(expected(cOp, ctest1)) ≈ 0. atol=1e-8
    @test imag(expected(cOp, ctest2)) ≈ 0. atol=1e-8
    @test imag(expected(cOp, ctest4)) ≈ 0. atol=1e-8
    @test imag(expected(cOp, ctest5)) ≈ 0. atol=1e-8

    # Test with complex hermitian Mpo.
    cOp = Mpo(ComplexF64, L, d)
    add_ops!(cOp, "c+", "c", cJ, ferm_op="Z")
    @test imag(expected(cOp, ctest1)) ≈ 0. atol=1e-8
    @test imag(expected(cOp, ctest2)) ≈ 0. atol=1e-8
    @test imag(expected(cOp, ctest4)) ≈ 0. atol=1e-8
    @test imag(expected(cOp, ctest5)) ≈ 0. atol=1e-8

    cOp = Mpo(ComplexF64, L, d)
    add_ops!(cOp, "c+", "c", cJ.^2)
    @test imag(expected(cOp, ctest1)) ≈ 0. atol=1e-8
    @test imag(expected(cOp, ctest2)) ≈ 0. atol=1e-8
    @test imag(expected(cOp, ctest4)) ≈ 0. atol=1e-8
    @test imag(expected(cOp, ctest5)) ≈ 0. atol=1e-8

    cOp = Mpo(ComplexF64, L, d)
    add_ops!(cOp, "n", "n", cJ)
    @test imag(expected(cOp, ctest1)) ≈ 0. atol=1e-8
    @test imag(expected(cOp, ctest2)) ≈ 0. atol=1e-8
    @test imag(expected(cOp, ctest4)) ≈ 0. atol=1e-8
    @test imag(expected(cOp, ctest5)) ≈ 0. atol=1e-8

    # Test with spin operators.
    d = 3
    Op = Mpo(L, d)
    add_ops!(Op, "Sz", "Sz", J)
    @test imag(expected(Op, rtest3)) ≈ 0. atol=1e-15
    Op = Mpo(L, d)
    add_ops!(Op, "S+", "S-", J.^2)
    @test imag(expected(Op, rtest3)) ≈ 0. atol=1e-15
    Op = Mpo(L, d)
    add_ops!(Op, "Sz", "Sz", J)
    add_ops!(Op, "S-", "S+", tan.(J))
    @test imag(expected(Op, rtest3)) ≈ 0. atol=1e-15
end

@testset "Contraction of bond dimensions of an Mpo" begin
    L = 10
    d = 2

    # Build an Mpo with first neighbor interactions, bond dimension cannot be
    # greater than 4.
    J = diagm(1 => rand(L-1)) + diagm(-1 => rand(L-1))
    Op = Mpo(L, d)
    add_ops!(Op, "b+", "b", J)
    @test maximum(size.(Op.W, 4)) <= 4

    # Add periodic bond dimensions, this should increase the bond dim to 5.
    Jpbc = zeros(L, L)
    Jpbc[1, L] = 1.
    Jpbc[L, 2] = 3.
    add_ops!(Op, "b+", "b", Jpbc)
    @test maximum(size.(Op.W, 4)) <= 6
end
end # @testset "Operations with Mpo"
