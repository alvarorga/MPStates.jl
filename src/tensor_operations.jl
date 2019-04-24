#
# Operations between tensors.
#

"""
    prop_qr(R::Array{T, 2}, A::Array{T, 3}, return_Q=true) where T<:Number

From two tensors of rank 2 and 3: -R-A-, contract and perform QR decomposition,
returning the equivalent tensors: -Q-R2, with ranks 3 and 2.
"""
function prop_qr(R::Array{T, 2}, A::Array{T, 3}, return_Q=true) where T<:Number
    @tensor RA[i, s, j] := R[i, k]*A[k, s, j]
    RA = reshape(RA, (size(RA, 1)*size(RA, 2), size(RA, 3)))
    rQ, R2 = qr(RA)
    return_Q || return R2
    rQ = Matrix(rQ)
    Q = reshape(rQ, (size(R, 1), size(A, 2), size(rQ, 2)))
    return Q, R2
end

"""
    prop_lq(A::Array{T, 3}, L::Array{T, 2}, return_Q=true) where T<:Number

From two tensors of rank 3 and 2: -A-L-, contract and perform LQ decomposition,
returning the equivalent tensors: -L2-Q-, with ranks 2 and 3.
"""
function prop_lq(A::Array{T, 3}, L::Array{T, 2}, return_Q=true) where T<:Number
    @tensor AL[i, s, j] := A[i, s, k]*L[k, j]
    AL = reshape(AL, (size(AL, 1), size(AL, 2)*size(AL, 3)))
    L2, rQ = lq(AL)
    return_Q || return L2
    rQ = Matrix(rQ)
    Q = reshape(rQ, (size(rQ, 1), size(A, 2), size(L, 2)))
    return L2, Q
end

"""
    prop_right2(L::Array{T, 2}, A::Array{T, 3}, B::Array{T, 3}) where T<:Number

Propagate the tensor L through the tensors A, B.
"""
function prop_right2(L::Array{T, 2}, A::Array{T, 3}, B::Array{T, 3}) where T<:Number
    @tensor begin
        L1[l1, s, b] := L[l1, l2]*conj(B[l2, s, b])
        new_L[a, b] := L1[l1, s, b]*A[l1, s, a]
    end
    return new_L
end

"""
    prop_right3(L::Array{T, 3}, A::Array{T, 3},
                M::Array{T, 4}, B::Array{T, 3}) where T<:Number

Propagate the tensor L through the tensors A, M, B.
"""
function prop_right3(L::Array{T, 3}, A::Array{T, 3},
                     M::Array{T, 4}, B::Array{T, 3}) where T<:Number
    @tensor begin
        L1[l1, l2, s2, c] := L[l1, l2, l3]*conj(B[l3, s2, c])
        L2[l1, s1, b, c] := L1[l1, l2, s2, c]*M[l2, s1, s2, b]
        new_L[a, b, c] := L2[l1, s1, b, c]*A[l1, s1, a]
    end
    return new_L
end

"""
    prop_right4(L::Array{T, 4}, A::Array{T, 3}, M1::Array{T, 4},
                M2::Array{T, 4}, B::Array{T, 3}) where T<:Number

Propagate the tensor L through the tensors A, M1, M2, B.
"""
function prop_right4(L::Array{T, 4}, A::Array{T, 3}, M1::Array{T, 4}, 
                     M2::Array{T, 4}, B::Array{T, 3}) where T<:Number
    @tensor begin
        L1[l1, l2, l3, s3, d] := L[l1, l2, l3, l4]*conj(B[l4, s3, d])
        L2[l1, l2, s2, c, d] := L1[l1, l2, l3, s3, d]*M2[l3, s2, s3, c]
        L3[l1, s1, b, c, d] := L2[l1, l2, s2, c, d]*M1[l2, s1, s2, b]
        new_L[a, b, c, d] := L3[l1, s1, b, c, d]*A[l1, s1, a]
    end
    return new_L
end
