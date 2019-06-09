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

"""
    apply!(Op::Mpo{T}, psi::Mps{T}) where T<:Number

Apply an Mpo to an Mps: Op|psi>. Keep new state unnormalized.
"""
function apply!(Op::Mpo{T}, psi::Mps{T}) where T<:Number
    psi.L == Op.L || throw("Mpo and Mps have different lengths.")
    psi.d == Op.d || throw("Mpo and Mps have different physical dimension.")
    for i=1:psi.L
        @tensor Ai[l1, l2, s2, r1, r2] := psi.A[i][l1, s1, r1]*Op.M[i][l2, s1, s2, r2]
        psi.A[i] = reshape(Ai, size(psi.A[i], 1)*size(Op.M[i], 1), psi.d,
                               size(psi.A[i], 3)*size(Op.M[i], 4))
    end
    norm_psi = norm(psi)
    # Make tensors left and right canonical. Make it twice so that bond
    # dimensions at the start and ending sites follow the rule 2^site.
    M = make_left_canonical(make_right_canonical(psi.A))
    left_can_A = M
    right_can_A = make_right_canonical(M)
    for i=1:psi.L
        psi.A[i] = deepcopy(left_can_A[i])
        psi.B[i] = deepcopy(right_can_A[i])
    end
    # Making left and right canonical normalizes the state, remove the norm.
    psi.A[end] .*= sqrt(norm_psi)
    psi.B[end] .*= sqrt(norm_psi)
    return
end
