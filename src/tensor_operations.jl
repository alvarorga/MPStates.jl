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
