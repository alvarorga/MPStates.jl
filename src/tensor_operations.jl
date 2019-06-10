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
    prop_right_svd(SVt::Array{T, 2}, A::Array{T, 3}, max_D::Int=-1) where T<:Number

From two tensors of rank 2 and 3: -SVt-A-, contract and perform SVD
decomposition, returning the equivalent tensors: -U-SVt2-, with ranks 3 and 2.
Also truncate bond dimension if maximum is reached.
"""
function prop_right_svd(SVt::Array{T, 2}, A::Array{T, 3}, max_D::Int=-1) where T<:Number
    @tensor new_A[i, s, j] := SVt[i, k]*A[k, s, j]
    new_A = reshape(new_A, (size(new_A, 1)*size(new_A, 2), size(new_A, 3)))
    F = svd(new_A)
    if max_D > 0 && length(F.S) > max_D
        U = F.U[:, 1:max_D]
        # Normalize state.
        S = Diagonal(F.S[1:max_D])/norm(F.S[1:max_D])
        Vt = F.Vt[1:max_D, :]
    else
        U = F.U
        S = Diagonal(F.S)
        Vt = F.Vt
    end
    SVt2 = S*Vt
    U = reshape(U, (size(SVt, 1), size(A, 2), size(SVt2, 1)))
    return U, SVt2
end

"""
    prop_left_svd(A::Array{T, 3}, US::Array{T, 2}, max_D::Int=-1) where T<:Number

From two tensors of rank 3 and 2: -A-US-, contract and perform SVD
decomposition, returning the equivalent tensors: -US2-Vt-, with ranks 2 and 3.
Also truncate bond dimension if maximum is reached.
"""
function prop_left_svd(A::Array{T, 3}, US::Array{T, 2}, max_D::Int=-1) where T<:Number
    @tensor new_A[i, s, j] := A[i, s, k]*US[k, j]
    new_A = reshape(new_A, (size(new_A, 1), size(new_A, 2)*size(new_A, 3)))
    F = svd(new_A)
    if max_D > 0 && length(F.S) > max_D
        U = F.U[:, 1:max_D]
        S = Diagonal(F.S[1:max_D])
        Vt = F.Vt[1:max_D, :]
    else
        U = F.U
        S = Diagonal(F.S)
        Vt = F.Vt
    end
    US2 = U*S
    Vt = reshape(Vt, (size(US2, 2), size(A, 2), size(US, 2)))
    return US2, Vt
end

"""
    prop_right2(L::Array{Float64, 2}, A::Array{Float64, 3}, B::Array{Float64, 3})

Propagate the tensor L through the tensors A, B.
"""
function prop_right2(L::Array{Float64, 2}, A::Array{Float64, 3}, B::Array{Float64, 3})
    Ar = reshape(A, size(A, 1), size(A, 2)*size(A, 3))
    T1 = BLAS.gemm('T', 'N', L, Ar)
    T1 = reshape(T1, size(T1, 1)*size(B, 2), size(A, 3))
    Br = reshape(B, size(B, 1)*size(B, 2), size(B, 3))
    L2 = BLAS.gemm('T', 'N', T1, Br)
    return L2
end

"""
    prop_right2(L::Array{ComplexF64, 2}, A::Array{ComplexF64, 3}, B::Array{ComplexF64, 3})

Propagate the tensor L through the tensors A, B.
"""
function prop_right2(L::Array{ComplexF64, 2}, A::Array{ComplexF64, 3}, B::Array{ComplexF64, 3})
    Ar = reshape(A, size(A, 1), size(A, 2)*size(A, 3))
    T1 = BLAS.gemm('T', 'N', L, Ar)
    T1 = reshape(T1, size(T1, 1)*size(B, 2), size(A, 3))
    Br = reshape(B, size(B, 1)*size(B, 2), size(B, 3))
    L2 = BLAS.gemm('T', 'N', T1, conj(Br))
    return L2
end

"""
    prop_right3(L::Array{T1, 3}, A::Array{T2, 3},
                M::Array{T3, 4}, B::Array{T4, 3}) where {T1, T2, T3, T4}

Propagate the tensor L through the tensors A, M, B.
"""
function prop_right3(L::Array{T1, 3}, A::Array{T2, 3},
                     M::Array{T3, 4}, B::Array{T4, 3}) where {T1, T2, T3, T4}
    @tensor begin
        L1[l1, l2, s2, c] := L[l1, l2, l3]*conj(B[l3, s2, c])
        L2[l1, s1, b, c] := L1[l1, l2, s2, c]*M[l2, s1, s2, b]
        new_L[a, b, c] := L2[l1, s1, b, c]*A[l1, s1, a]
    end
    return new_L
end

"""
    prop_right4(L::Array{T1, 4}, A::Array{T2, 3}, M1::Array{T3, 4},
                M2::Array{T4, 4}, B::Array{T5, 3}) where {T1, T2, T3, T4, T5}

Propagate the tensor L through the tensors A, M1, M2, B.
"""
function prop_right4(L::Array{T1, 4}, A::Array{T2, 3}, M1::Array{T3, 4},
                     M2::Array{T4, 4}, B::Array{T5, 3}) where {T1, T2, T3, T4, T5}
    @tensor begin
        L1[l1, l2, l3, s3, d] := L[l1, l2, l3, l4]*conj(B[l4, s3, d])
        L2[l1, l2, s2, c, d] := L1[l1, l2, l3, s3, d]*M2[l3, s2, s3, c]
        L3[l1, s1, b, c, d] := L2[l1, l2, s2, c, d]*M1[l2, s1, s2, b]
        new_L[a, b, c, d] := L3[l1, s1, b, c, d]*A[l1, s1, a]
    end
    return new_L
end

"""
    prop_left2(A::Array{Float64, 3}, B::Array{Float64, 3}, R::Array{Float64, 2})

Propagate the tensor R through the tensors A, B.
"""
function prop_left2(A::Array{Float64, 3}, B::Array{Float64, 3}, R::Array{Float64, 2})
    Ar = reshape(A, size(A, 1)*size(A, 2), size(A, 3))
    T1 = BLAS.gemm('N', 'N', Ar, R)
    T1 = reshape(T1, size(A, 1), size(A, 2)*size(T1, 2))
    Br = reshape(B, size(B, 1), size(B, 2)*size(B, 3))
    L2 = BLAS.gemm('N', 'T', T1, Br)
    return L2
end

"""
    prop_left2(A::Array{ComplexF64, 3}, B::Array{ComplexF64, 3}, R::Array{ComplexF64, 2})

Propagate the tensor R through the tensors A, B.
"""
function prop_left2(A::Array{ComplexF64, 3}, B::Array{ComplexF64, 3}, R::Array{ComplexF64, 2})
    Ar = reshape(A, size(A, 1)*size(A, 2), size(A, 3))
    T1 = BLAS.gemm('N', 'N', Ar, R)
    T1 = reshape(T1, size(A, 1), size(A, 2)*size(T1, 2))
    Br = reshape(B, size(B, 1), size(B, 2)*size(B, 3))
    L2 = BLAS.gemm('N', 'C', T1, Br)
    return L2
end

"""
    prop_left3(A::Array{T1, 3}, M::Array{T2, 4}, B::Array{T3, 3},
               R::Array{T4, 3}) where {T1, T2, T3, T4}

Propagate the tensor R through the tensors A, M, B.
"""
function prop_left3(A::Array{T1, 3}, M::Array{T2, 4}, B::Array{T3, 3},
                    R::Array{T4, 3}) where {T1, T2, T3, T4}
    @tensor begin
        R1[a, s1, r2, r3] := A[a, s1, r1]*R[r1, r2, r3]
        R2[a, b, s2, r3] := R1[a, s1, r2, r3]*M[b, s1, s2, r2]
        new_R[a, b, c] := R2[a, b, s2, r3]*conj(B[c, s2, r3])
    end
    return new_R
end

"""
    prop_left4(A::Array{T1, 3}, M1::Array{T2, 4}, M2::Array{T3, 4},
               B::Array{T4, 3}, R::Array{T5, 4}) where {T1, T2, T3, T4, T5}

Propagate the tensor R through the tensors A, M1, M2, B.
"""
function prop_left4(A::Array{T1, 3}, M1::Array{T2, 4}, M2::Array{T3, 4},
                    B::Array{T4, 3}, R::Array{T5, 4}) where {T1, T2, T3, T4, T5}
    @tensor begin
        R1[a, s1, r2, r3, r4] := A[a, s1, r1]*R[r1, r2, r3, r4]
        R2[a, b, s2, r3, r4] := R1[a, s1, r2, r3, r4]*M1[b, s1, s2, r2]
        R3[a, b, c, s3, r4] := R2[a, b, s2, r3, r4]*M2[c, s2, s3, r3]
        new_R[a, b, c, d] := R3[a, b, c, s3, r4]*conj(B[d, s3, r4])
    end
    return new_R
end
