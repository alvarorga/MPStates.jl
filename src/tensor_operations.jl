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
    prop_right2(L::Array{T, 2}, A::Array{T, 3}, B::Array{T, 3}) where T<:Number

Propagate the tensor L through the tensors A, B.
"""
function prop_right2(L::Array{T, 2}, A::Array{T, 3}, B::Array{T, 3}) where T<:Number
    Ar = reshape(A, size(A, 1), size(A, 2)*size(A, 3))
    T1 = BLAS.gemm('T', 'N', L, Ar)
    T1 = reshape(T1, size(T1, 1)*size(B, 2), size(A, 3))
    Br = reshape(B, size(B, 1)*size(B, 2), size(B, 3))
    L2 = BLAS.gemm('T', 'N', T1, conj(Br))
    return L2
end

"""
    prop_right3(L::Array{T, 3}, A::Array{T, 3}, M::Array{T, 4}, B::Array{T, 3}) where T<:Number

Propagate the tensor L through the tensors A, M, B.
"""
function prop_right3(L::Array{T, 3}, A::Array{T, 3}, M::Array{T, 4}, B::Array{T, 3}) where T<:Number
    @tensoropt begin
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
    @tensoropt begin
        L1[l1, l2, l3, s3, d] := L[l1, l2, l3, l4]*conj(B[l4, s3, d])
        L2[l1, l2, s2, c, d] := L1[l1, l2, l3, s3, d]*M2[l3, s2, s3, c]
        L3[l1, s1, b, c, d] := L2[l1, l2, s2, c, d]*M1[l2, s1, s2, b]
        new_L[a, b, c, d] := L3[l1, s1, b, c, d]*A[l1, s1, a]
    end
    return new_L
end

"""
    prop_left2(A::Array{T, 3}, B::Array{T, 3}, R::Array{T, 2}) where T<:Number

Propagate the tensor R through the tensors A, B.
"""
function prop_left2(A::Array{T, 3}, B::Array{T, 3}, R::Array{T, 2}) where T<:Number
    Ar = reshape(A, size(A, 1)*size(A, 2), size(A, 3))
    T1 = BLAS.gemm('N', 'N', Ar, R)
    T1 = reshape(T1, size(A, 1), size(A, 2)*size(T1, 2))
    Br = reshape(B, size(B, 1), size(B, 2)*size(B, 3))
    L2 = BLAS.gemm('N', 'C', T1, Br)
    return L2
end

"""
    prop_left3(A::Array{T, 3}, M::Array{T, 4}, B::Array{T, 3}, R::Array{T, 3}) where T<:Number

Propagate the tensor R through the tensors A, M, B.
"""
function prop_left3(A::Array{T, 3}, M::Array{T, 4}, B::Array{T, 3}, R::Array{T, 3}) where T<:Number
    @tensoropt new_R[a, b, c] := A[a, s1, r1]*M[b, s1, s2, r2]*conj(B[c, s2, r3])*R[r1, r2, r3]
    return new_R
end

"""
    prop_left4(A::Array{T, 3}, M1::Array{T, 4}, M2::Array{T, 4},
               B::Array{T, 3}, R::Array{T, 4}) where T<:Number

Propagate the tensor R through the tensors A, M1, M2, B.
"""
function prop_left4(A::Array{T, 3}, M1::Array{T, 4}, M2::Array{T, 4},
                    B::Array{T, 3}, R::Array{T, 4}) where T<:Number
    @tensoropt begin
        R1[a, s1, r2, r3, r4] := A[a, s1, r1]*R[r1, r2, r3, r4]
        R2[a, b, s2, r3, r4] := R1[a, s1, r2, r3, r4]*M1[b, s1, s2, r2]
        R3[a, b, c, s3, r4] := R2[a, b, s2, r3, r4]*M2[c, s2, s3, r3]
        new_R[a, b, c, d] := R3[a, b, c, s3, r4]*conj(B[d, s3, r4])
    end
    return new_R
end

"""
    prop_right_subexp(L::Array{T, 3}, W::Array{T, 4}, M::Array{T, 3}) where T<:Number

Propagate the tensor L through the tensors W, M for the DMRG3S algorithm.
"""
function prop_right_subexp(L::Array{T, 3}, W::Array{T, 4}, M::Array{T, 3}) where T<:Number
    @tensoropt P[l1, s1, r2, r3] := L[l1, l2, l3]*conj(M[l3, s2, r3])*W[l2, s1, s2, r2]
    return P
end

"""
    prop_left_subexp(W::Array{T, 4}, M::Array{T, 3}, R::Array{T, 3}) where T<:Number

Propagate the tensor R through the tensors W, M for the DMRG3S algorithm.
"""
function prop_left_subexp(W::Array{T, 4}, M::Array{T, 3}, R::Array{T, 3}) where T<:Number
    @tensoropt begin
        R1[r1, r2, s2, l3] := R[r1, r2, r3]*conj(M[l3, s2, r3])
        P[s1, r1, l2, l3] := R1[r1, r2, s2, l3]*W[l2, s1, s2, r2]
    end
    return P
end

"""
    absorb_Le(Le::Array{T, 3}, M::Array{T, 2}) where T<:Number

Absorb M into the left environment Le.
"""
function absorb_Le(Le::Array{T, 3}, M::Matrix{T}) where T<:Number
    if size(M, 1) == size(M, 2)
        @tensoropt Le[l1, l2, l3] = Le[a, l2, b]*M[a, l1]*conj(M[b, l3])
    else
        @tensoropt Le[l1, l2, l3] := Le[a, l2, b]*M[a, l1]*conj(M[b, l3])
    end
    return Le
end

"""
    absorb_Re(Re::Array{T, 3}, M::Array{T, 2}) where T<:Number

Absorb M into the right environment Re.
"""
function absorb_Re(Re::Array{T, 3}, M::Matrix{T}) where T<:Number
    if size(M, 1) == size(M, 2)
        @tensoropt Re[r1, r2, r3] = M[r1, a]*conj(M[r3, b])*Re[a, r2, b]
    else
        @tensoropt Re[r1, r2, r3] := M[r1, a]*conj(M[r3, b])*Re[a, r2, b]
    end
    return Re
end

"""
    build_local_hamiltonian(Le::Array{T, 3}, W::Array{T, 4},
                            Re::Array{T, 3}) where T<:Number

Build the local Hamiltonian with the left and right environments.
"""
function build_local_hamiltonian(Le::Array{T, 3}, W::Array{T, 4},
                                 Re::Array{T, 3})  where T<:Number
    @tensoropt Hi[l1, s1, r1, l3, s2, r3] := (Le[l1, l2, l3]
                                              *W[l2, s1, s2, r2]
                                              *Re[r1, r2, r3])
    Hi = reshape(Hi, (size(Le, 1)*size(W, 2)*size(Re, 1),
                      size(Le, 3)*size(W, 3)*size(Re, 3)))
    return Hi
end

"""
    build_local_hamiltonian(Le::Array{T, 3}, W::Array{T, 4}, Re::Array{T, 3},
                            cache::Cache{T}) where T<:Number

Build the local Hamiltonian with the left and right environments. Pass a
preallocated cache that can be used to store the final tensor.
"""
function build_local_hamiltonian(Le::Array{T, 3}, W::Array{T, 4}, Re::Array{T, 3},
                                 cache::Cache{T})  where T<:Number
    # Check if any cache element of appropriate size can be used.
    needed_space = size(Le, 1)*size(W, 2)*size(Re, 1)*size(Le, 3)*size(W, 3)*size(Re, 3)
    loc_in_cache = is_in_cache(cache, needed_space)
    if loc_in_cache > 0
        Hi = reshape(cache.elts[loc_in_cache],
                     (size(Le, 1), size(W, 2), size(Re, 1),
                      size(Le, 3), size(W, 3), size(Re, 3)))
        @tensoropt Hi[l1, s1, r1, l3, s2, r3] = (Le[l1, l2, l3]
                                                 *W[l2, s1, s2, r2]
                                                 *Re[r1, r2, r3])
    else
        @tensoropt Hi[l1, s1, r1, l3, s2, r3] := (Le[l1, l2, l3]
                                                  *W[l2, s1, s2, r2]
                                                  *Re[r1, r2, r3])
    end
    Hi = reshape(Hi, (size(Le, 1)*size(W, 2)*size(Re, 1),
                      size(Le, 3)*size(W, 3)*size(Re, 3)))
    update_cache!(cache, Hi)
    return Hi
end

"""
    build_local_hamiltonian_2(Le::Array{T, 3}, W1::Array{T, 4},
                              W2::Array{T, 4}, Re::Array{T, 3},
                              cache::Cache{T}) where T<:Number

Build the local Hamiltonian with the left and right environments for DMRG2..
"""
function build_local_hamiltonian_2(Le::Array{T, 3}, W1::Array{T, 4},
                                   W2::Array{T, 4}, Re::Array{T, 3},
                                   cache::Cache{T}) where T<:Number
    @tensoropt Hi[l1, s1, s3, r1, l3, s2, s4, r3] := (Le[l1, l2, l3]
                                                      *W1[l2, s1, s2, a]
                                                      *W2[a, s3, s4, r2]
                                                      *Re[r1, r2, r3])
    Hi = reshape(Hi, (size(Le, 1)*size(W1, 2)*size(W2, 2)*size(Re, 1),
                      size(Le, 3)*size(W1, 3)*size(W2, 3)*size(Re, 3)))
    # Check if any cache element of appropriate size can be used.
    needed_space = (size(Le, 1)*size(W1, 2)*size(W2, 2)*size(Re, 1)
                    *size(Le, 3)*size(W1, 3)*size(W2, 3)*size(Re, 3))
    loc_in_cache = is_in_cache(cache, needed_space)
    if loc_in_cache > 0
        Hi = reshape(cache.elts[loc_in_cache],
                     ((size(Le, 1), size(W1, 2), size(W2, 2), size(Re, 1),
                      size(Le, 3),size(W1, 3),size(W2, 3),size(Re, 3))))
        @tensoropt Hi[l1, s1, s3, r1, l3, s2, s4, r3] = (Le[l1, l2, l3]
                                                         *W1[l2, s1, s2, a]
                                                         *W2[a, s3, s4, r2]
                                                         *Re[r1, r2, r3])
    else
        @tensoropt Hi[l1, s1, s3, r1, l3, s2, s4, r3] := (Le[l1, l2, l3]
                                                          *W1[l2, s1, s2, a]
                                                          *W2[a, s3, s4, r2]
                                                          *Re[r1, r2, r3])
    end
    Hi = reshape(Hi, ((size(Le, 1)*size(W1, 2)*size(W2, 2)*size(Re, 1),
                       size(Le, 3)*size(W1, 3)*size(W2, 3)*size(Re, 3))))
    update_cache!(cache, Hi)
    return Hi
end
