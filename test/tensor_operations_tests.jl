@testset "Tensor operations" begin
@testset "right propagation of A and B" begin
    nl1 = 2
    nl2 = 3
    na = 4
    nb = 5
    ns = 6

    # Float64 propagation.
    L = reshape(collect(1. : nl1*nl2), (nl1, nl2))
    A = reshape(collect(1. : nl1*ns*na), (nl1, ns, na))
    B = reshape(collect(3. : 2+nl2*ns*nb), (nl2, ns, nb))
    # Do manual contraction first.
    new_L = zeros(ComplexF64, (na, nb))
    for a=1:na, b=1:nb
        for l1=1:nl1, l2=1:nl2, s=1:ns
            new_L[a, b] += L[l1, l2]*A[l1, s, a]*B[l2, s, b]
        end
    end
    @test MPStates.prop_right2(L, A, B) ≈ new_L

    # ComplexF64 propagation.
    L = reshape(complex.(collect(1. : nl1*nl2)), (nl1, nl2))
    A = reshape(complex.(collect(1. : nl1*ns*na), collect(-2. : -3+nl1*ns*na)),
                (nl1, ns, na))
    B = reshape(complex.(collect(3. : 2+nl2*ns*nb), collect(-1. : -2+nl2*ns*nb)),
                (nl2, ns, nb))
    # Do manual contraction first.
    new_L = zeros(ComplexF64, (na, nb))
    for a=1:na, b=1:nb
        for l1=1:nl1, l2=1:nl2, s=1:ns
            new_L[a, b] += L[l1, l2]*A[l1, s, a]*conj(B[l2, s, b])
        end
    end
    @test MPStates.prop_right2(L, A, B) ≈ new_L
end

@testset "right propagation of A, M, and B" begin
    nl1 = 2
    nl2 = 3
    nl3 = 4
    na = 4
    nb = 5
    nc = 3
    ns1 = 6
    ns2 = 2
    L = reshape(complex.(collect(1. : nl1*nl2*nl3)), (nl1, nl2, nl3))
    A = reshape(collect(1. : nl1*ns1*na) + 1im*collect(-2. : -3+nl1*ns1*na),
                (nl1, ns1, na))
    M = reshape(collect(1. : nl2*ns1*ns2*nb) + 1im*collect(0. : -1+nl2*ns1*ns2*nb),
                (nl2, ns1, ns2, nb))
    B = reshape(collect(3. : 2+nl3*ns2*nc) + 1im*collect(-1. : -2+nl3*ns2*nc),
                (nl3, ns2, nc))
    # Do manual contraction first.
    new_L = zeros(ComplexF64, (na, nb, nc))
    for a=1:na, b=1:nb, c=1:nc
        for l1=1:nl1, l2=1:nl2, l3=1:nl3, s1=1:ns1, s2=1:ns2
            new_L[a, b, c] += L[l1, l2, l3]*A[l1, s1, a]*M[l2, s1, s2, b]*conj(B[l3, s2, c])
        end
    end
    @test MPStates.prop_right3(L, A, M, B) ≈ new_L
end

@testset "right propagation of M, and B for DMRG3S" begin
    nl1 = 2
    nl2 = 3
    nl3 = 4
    nr2 = 5
    nr3 = 3
    ns1 = 6
    ns2 = 2
    L = reshape(complex.(collect(1. : nl1*nl2*nl3)), (nl1, nl2, nl3))
    M = reshape(complex.(collect(1. : nl2*ns1*ns2*nr2), collect(0. : -1+nl2*ns1*ns2*nr2)),
                (nl2, ns1, ns2, nr2))
    B = reshape(complex.(collect(3. : 2+nl3*ns2*nr3), collect(-1. : -2+nl3*ns2*nr3)),
                (nl3, ns2, nr3))
    # Do manual contraction first.
    P = zeros(ComplexF64, (nl1, ns1, nr2, nr3))
    for l1=1:nl1, s1=1:ns1, r2=1:nr2, r3=1:nr3
        for l2=1:nl2, l3=1:nl3, s2=1:ns2
            P[l1, s1, r2, r3] += L[l1, l2, l3]*M[l2, s1, s2, r2]*conj(B[l3, s2, r3])
        end
    end
    @test MPStates.prop_right_subexp(L, M, B) ≈ P
end

@testset "right propagation of A, M1, M2, and B" begin
    nl1 = 2
    nl2 = 3
    nl3 = 4
    nl4 = 4
    na = 4
    nb = 5
    nc = 3
    nd = 5
    ns1 = 6
    ns2 = 2
    ns3 = 4
    L = reshape(complex.(collect(1. : nl1*nl2*nl3*nl4)), (nl1, nl2, nl3, nl4))
    A = reshape(collect(1. : nl1*ns1*na) + 1im*collect(-2. : -3+nl1*ns1*na),
                (nl1, ns1, na))
    M1 = reshape(collect(1. : nl2*ns1*ns2*nb) + 1im*collect(0. : -1+nl2*ns1*ns2*nb),
                 (nl2, ns1, ns2, nb))
    M2 = reshape(collect(1. : nl3*ns2*ns3*nc) + 1im*collect(0. : -1+nl3*ns2*ns3*nc),
                 (nl3, ns2, ns3, nc))
    B = reshape(collect(3. : 2+nl4*ns3*nd) + 1im*collect(-1. : -2+nl4*ns3*nd),
                (nl4, ns3, nd))
    # Do manual contraction first.
    new_L = zeros(ComplexF64, (na, nb, nc, nd))
    for a=1:na, b=1:nb, c=1:nc, d=1:nd
        for l1=1:nl1, l2=1:nl2, l3=1:nl3, l4=1:nl4, s1=1:ns1, s2=1:ns2, s3=1:ns3
            new_L[a, b, c, d] += (L[l1, l2, l3, l4]*A[l1, s1, a]*M1[l2, s1, s2, b]
                                  *M2[l3, s2, s3, c]*conj(B[l4, s3, d]))
        end
    end
    @test MPStates.prop_right4(L, A, M1, M2, B) ≈ new_L
end

@testset "left propagation of A and B" begin
    nr1 = 2
    nr2 = 3
    na = 4
    nb = 5
    ns = 6

    # Float64 propagation.
    R = reshape(collect(1. : nr1*nr2), (nr1, nr2))
    A = reshape(collect(1. : nr1*ns*na), (na, ns, nr1))
    B = reshape(collect(3. : 2+nr2*ns*nb), (nb, ns, nr2))
    # Do manual contraction first.
    new_R = zeros(ComplexF64, (na, nb))
    for a=1:na, b=1:nb
        for r1=1:nr1, r2=1:nr2, s=1:ns
            new_R[a, b] += R[r1, r2]*A[a, s, r1]*B[b, s, r2]
        end
    end
    @test MPStates.prop_left2(A, B, R) ≈ new_R

    # ComplexF64 propagation.
    R = reshape(complex.(collect(1. : nr1*nr2)), (nr1, nr2))
    A = reshape(complex.(collect(1. : nr1*ns*na), collect(-2. : -3+nr1*ns*na)),
                (na, ns, nr1))
    B = reshape(complex.(collect(3. : 2+nr2*ns*nb), collect(-1. : -2+nr2*ns*nb)),
                (nb, ns, nr2))
    # Do manual contraction first.
    new_R = zeros(ComplexF64, (na, nb))
    for a=1:na, b=1:nb
        for r1=1:nr1, r2=1:nr2, s=1:ns
            new_R[a, b] += R[r1, r2]*A[a, s, r1]*conj(B[b, s, r2])
        end
    end
    @test MPStates.prop_left2(A, B, R) ≈ new_R
end

@testset "left propagation of A, M, and B" begin
    nr1 = 2
    nr2 = 3
    nr3 = 4
    na = 4
    nb = 5
    nc = 3
    ns1 = 6
    ns2 = 2
    R = reshape(complex.(collect(1. : nr1*nr2*nr3)), (nr1, nr2, nr3))
    A = reshape(collect(1. : nr1*ns1*na) + 1im*collect(-2. : -3+nr1*ns1*na),
                (na, ns1, nr1))
    M = reshape(collect(1. : nr2*ns1*ns2*nb) + 1im*collect(0. : -1+nr2*ns1*ns2*nb),
                (nb, ns1, ns2, nr2))
    B = reshape(collect(3. : 2+nr3*ns2*nc) + 1im*collect(-1. : -2+nr3*ns2*nc),
                (nc, ns2, nr3))
    # Do manual contraction first.
    new_R = zeros(ComplexF64, (na, nb, nc))
    for a=1:na, b=1:nb, c=1:nc
        for r1=1:nr1, r2=1:nr2, r3=1:nr3, s1=1:ns1, s2=1:ns2
            new_R[a, b, c] += R[r1, r2, r3]*A[a, s1, r1]*M[b, s1, s2, r2]*conj(B[c, s2, r3])
        end
    end
    @test MPStates.prop_left3(A, M, B, R) ≈ new_R
end

@testset "left propagation of M, and B for DMRG3S" begin
    nr1 = 2
    nr2 = 3
    nr3 = 4
    nl2 = 5
    nl3 = 3
    ns1 = 6
    ns2 = 2
    R = reshape(complex.(collect(1. : nr1*nr2*nr3)), (nr1, nr2, nr3))
    M = reshape(complex.(collect(1. : nr2*ns1*ns2*nl2), (0. : -1+nr2*ns1*ns2*nl2)),
                (nl2, ns1, ns2, nr2))
    B = reshape(complex.(collect(3. : 2+nr3*ns2*nl3), (-1. : -2+nr3*ns2*nl3)),
                (nl3, ns2, nr3))
    # Do manual contraction first.
    P = zeros(ComplexF64, (ns1, nr1, nl2, nl3))
    for r1=1:nr1, s1=1:ns1, l2=1:nl2, l3=1:nl3
        for r2=1:nr2, r3=1:nr3, s2=1:ns2
            P[s1, r1, l2, l3] += R[r1, r2, r3]*M[l2, s1, s2, r2]*conj(B[l3, s2, r3])
        end
    end
    @test MPStates.prop_left_subexp(M, B, R) ≈ P
end

@testset "left propagation of A, M1, M2, and B" begin
    nr1 = 2
    nr2 = 3
    nr3 = 4
    nr4 = 4
    na = 4
    nb = 5
    nc = 3
    nd = 5
    ns1 = 6
    ns2 = 2
    ns3 = 4
    R = reshape(complex.(collect(1. : nr1*nr2*nr3*nr4)), (nr1, nr2, nr3, nr4))
    A = reshape(collect(1. : nr1*ns1*na) + 1im*collect(-2. : -3+nr1*ns1*na),
                (na, ns1, nr1))
    M1 = reshape(collect(1. : nr2*ns1*ns2*nb) + 1im*collect(0. : -1+nr2*ns1*ns2*nb),
                 (nb, ns1, ns2, nr2))
    M2 = reshape(collect(1. : nr3*ns2*ns3*nc) + 1im*collect(0. : -1+nr3*ns2*ns3*nc),
                 (nc, ns2, ns3, nr3))
    B = reshape(collect(3. : 2+nr4*ns3*nd) + 1im*collect(-1. : -2+nr4*ns3*nd),
                (nd, ns3, nr4))
    # Do manual contraction first.
    new_R = zeros(ComplexF64, (na, nb, nc, nd))
    for a=1:na, b=1:nb, c=1:nc, d=1:nd
        for r1=1:nr1, r2=1:nr2, r3=1:nr3, r4=1:nr4, s1=1:ns1, s2=1:ns2, s3=1:ns3
            new_R[a, b, c, d] += (R[r1, r2, r3, r4]*A[a, s1, r1]*M1[b, s1, s2, r2]
                                  *M2[c, s2, s3, r3]*conj(B[d, s3, r4]))
        end
    end
    @test MPStates.prop_left4(A, M1, M2, B, R) ≈ new_R
end
end # @testset "Tensor operations"
