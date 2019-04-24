@testset "right propagation of A and B" begin
    nl1 = 2
    nl2 = 3
    na = 4
    nb = 5
    ns = 6
    L = reshape(complex.(collect(1. : nl1*nl2)), (nl1, nl2))
    A = reshape(collect(1. : nl1*ns*na) + 1im*collect(-2. : -3+nl1*ns*na),
                (nl1, ns, na))
    B = reshape(collect(3. : 2+nl2*ns*nb) + 1im*collect(-1. : -2+nl2*ns*nb),
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
