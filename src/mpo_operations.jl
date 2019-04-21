#
# Operations between MPS and MPO.
#

"""
    expected(Op::Mpo, psi::Mps)

Compute expectation value < psi|Op|psi >.
"""
function expected(Op::Mpo{T}, psi::Mps{T}) where T
    L = ones(T, 1, 1, 1)
    for i=1:psi.L
        Ai = psi.A[i]
        Mi = Op.M[i]
        @tensor begin
            L1[a, s1, l2, l3] := L[l1, l2, l3]*Ai[l1, s1, a]
            L2[a, b, s2, l3] := L1[a, s1, l2, l3]*Mi[l2, s1, s2, b]
            L[a, b, c] := L2[a, b, s2, l3]*conj(Ai[l3, s2, c])
        end
    end
    return L[1, 1, 1]
end

"""
    expected2(Op::Mpo, psi::Mps)

Compute expectation value of the squared operator < psi|Op^2|psi >.
"""
function expected2(Op::Mpo{T}, psi::Mps{T}) where T
    L = ones(T, 1, 1, 1, 1)
    for i=1:psi.L
        Ai = psi.A[i]
        Mi = Op.M[i]
        @tensor begin
            L1[a, s1, l2, l3, l4] := L[l1, l2, l3, l4]*Ai[l1, s1, a]
            L2[a, b, s2, l3, l4] := L1[a, s1, l2, l3, l4]*Mi[l2, s1, s2, b]
            L3[a, b, c, s3, l4] := L2[a, b, s2, l3, l4]*Mi[l3, s2, s3, c]
            L[a, b, c, d] := L3[a, b, c, s3, l4]*conj(Ai[l4, s3, d])
        end
    end
    return L[1, 1, 1]
end
