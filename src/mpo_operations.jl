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
        L = prop_right3(L, psi.A[i], Op.M[i], psi.A[i])
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
        L = prop_right4(L, psi.A[i], Op.M[i], Op.M[i], psi.A[i])
    end
    return L[1, 1, 1, 1]
end

"""
    m_variance(Op::Mpo{T}, psi::Mps{T}) where T

Measure variance: <psi|Op^2|psi> - <psi|Op|psi>^2.
"""
function m_variance(Op::Mpo{T}, psi::Mps{T}) where T
    return expected2(Op, psi) - expected(Op, psi)^2
end
