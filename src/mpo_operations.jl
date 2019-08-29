export expected,
       m_variance,
       apply!

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
    norm_of_apply(Op::Mpo{T}, psi::Mps{T}) where T<:Number

Compute norm of the operator acting on the state: || Op|psi > ||^2.
"""
function norm_of_apply(Op::Mpo{T}, psi::Mps{T}) where T<:Number
    L = ones(T, 1, 1, 1, 1)
    for i=1:psi.L
        hOp = permutedims(conj(Op.W[i]), (1, 3, 2, 4))
        L = prop_right4(L, psi.M[i], Op.W[i], hOp, psi.M[i])
    end
    return real(L[1, 1, 1, 1])
end

"""
    m_variance(Op::Mpo{T}, psi::Mps{T}) where T<:Number

Measure variance: <psi|Op^2|psi> - |<psi|Op|psi>|^2, return a real number.
"""
function m_variance(Op::Mpo{T}, psi::Mps{T}) where T<:Number
    return norm_of_apply(Op, psi) - abs2(expected(Op, psi))
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

"""
    str_to_op(str_op::String)

Return the matrix that corresponds to an operator input as a string. Example:
"n" -> [[0. 0.];
        [0. 1.]].
"""
function str_to_op(str_op::String, d::Int=0)
    # Operators for 2 physical dimensions: fermions and hard-core bosons.
    if str_op == "n"
        # Number operator.
        return [[0. 0.];
                [0. 1.]]
    elseif str_op == "a+" || str_op == "b+" || str_op == "c+"
        # Creation operator.
        return [[0. 1.];
                [0. 0.]]
    elseif str_op == "a" || str_op == "b" || str_op == "c"
        # Creation operator.
        return [[0. 0.];
                [1. 0.]]
    elseif str_op == "Z"
        # Parity sign operator.
        return [[1. 0.];
                [0. -1.]]
    # Operators for 3 physical dimensions: spin 1 particles. State 1 is |1, -1>,
    # state 2 is |1, 0> and state 3 is |1, +1>, with |s, m>.
    elseif str_op == "Sz"
        return [[-1. 0. 0.];
                [0. 0. 0.];
                [0. 0. 1.]]
    elseif str_op == "S+"
        return [[0. sqrt(2) 0.];
                [0. 0. sqrt(2)];
                [0. 0. 0.]]
    elseif str_op == "S-"
        return [[0. 0. 0.];
                [sqrt(2) 0. 0.];
                [0. sqrt(2) 0.]]
    # General identity matrix with the same physical dimensions as the Mpo.
    elseif str_op == "Id"
        return Matrix{Float64}(I, d, d)
    else
        throw("Operator $str_op is not defined.")
    end

end
