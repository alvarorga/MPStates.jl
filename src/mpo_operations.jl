#
# Operations between MPS and MPO.
#

"""
    expected(Op::Mpo{T}, psi::Mps{T}) where T<:Number

Compute expectation value < psi|Op|psi >.
"""
function expected(Op::Mpo{T}, psi::Mps{T}) where T<:Number
    L = ones(T, 1, 1, 1)
    for i=1:psi.L
        L = prop_right3(L, psi.M[i], Op.W[i], psi.M[i])
    end
    return L[1, 1, 1]
end

"""
    expected2(Op::Mpo{T}, psi::Mps{T}) where T<:Number

Compute norm of the operator acting on the state: || Op|psi > ||^2.
"""
function expected2(Op::Mpo{T}, psi::Mps{T}) where T<:Number
    L = ones(T, 1, 1, 1, 1)
    for i=1:psi.L
        hOp = permutedims(conj(Op.W[i]), (1, 3, 2, 4))
        L = prop_right4(L, psi.M[i], Op.W[i], hOp, psi.M[i])
    end
    return L[1, 1, 1, 1]
end

"""
    m_variance(Op::Mpo{T}, psi::Mps{T}) where T<:Number

Measure variance: <psi|Op^2|psi> - <psi|Op|psi>^2.
"""
function m_variance(Op::Mpo{T}, psi::Mps{T}) where T<:Number
    return expected2(Op, psi) - expected(Op, psi)^2
end

"""
    apply!(Op::Mpo{T}, psi::Mps{T}) where T<:Number

Apply an Mpo to an Mps: Op|psi>. Keep new state unnormalized.
"""
function apply!(Op::Mpo{T}, psi::Mps{T}) where T<:Number
    psi.L == Op.L || throw("Mpo and Mps have different lengths.")
    psi.d == Op.d || throw("Mpo and Mps have different physical dimension.")
    for i=1:psi.L
        @tensor Mi[l1, l2, s2, r1, r2] := psi.M[i][l1, s1, r1]*Op.W[i][l2, s1, s2, r2]
        psi.M[i] = reshape(Mi, size(psi.M[i], 1)*size(Op.W[i], 1), psi.d,
                               size(psi.M[i], 3)*size(Op.W[i], 4))
    end
    return psi
end
