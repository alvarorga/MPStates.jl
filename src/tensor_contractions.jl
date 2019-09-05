"""
    absorb_fromleft(A::Array{T, 2}, B::Array{T, 3}) where T<:Number

Absorb the matrix A in B where A is at the left of B.

 -- A -- B -- = -- C --
         |         |

"""
function absorb_fromleft(A::Array{T, 2}, B::Array{T, 3}) where T<:Number
    a1, a2 = size(A)
    b1, b2, b3 = size(B)
    B = reshape(B, b1, b2*b3)
    return reshape(A*B, a1, b2, b3)
end

"""
    absorb_fromright(A::Array{T, 3}, B::Array{T, 2}) where T<:Number

Absorb the matrix B in A where B is at the right of A.

 -- A -- B -- = -- C --
    |              |

"""
function absorb_fromright(A::Array{T, 3}, B::Array{T, 2}) where T<:Number
    a1, a2, a3 = size(A)
    b1, b2 = size(B)
    A = reshape(A, a1*a2, a3)
    return reshape(A*B, a1, a2, b2)
end

"""
    prop_right2(L::Array{T, 2}, A::Array{T, 3}, B::Array{T, 3}) where T<:Number

Propagate the tensor L through the tensors A, conj(B).

 ---- A ---     ---
 |    |         |
 L    |     =  L2
 |    |         |
 ---- B ---     ---

"""
function prop_right2(L::Array{T, 2}, A::Array{T, 3},
                     B::Array{T, 3}) where T<:Number
    Ar = reshape(A, size(A, 1), size(A, 2)*size(A, 3))
    T1 = BLAS.gemm('T', 'N', L, Ar)
    T1 = reshape(T1, size(T1, 1)*size(B, 2), size(A, 3))
    Br = reshape(B, size(B, 1)*size(B, 2), size(B, 3))
    L2 = BLAS.gemm('T', 'N', T1, conj(Br))
    return L2
end

"""
    prop_right3(L::Array{T, 3}, A::Array{T, 3}, M::Array{T, 4},
                B::Array{T, 3}) where T<:Number

Propagate the tensor L through the tensors A, M, conj(B).

 ---- A ---         ---
 |    |             |
 L--- M ---  =  new_L
 |    |             |
 ---- B ---         ---

"""
function prop_right3(L::Array{T, 3}, A::Array{T, 3}, M::Array{T, 4},
                     B::Array{T, 3}) where T<:Number
    @tensoropt L1[l1, l2, s2, c] := L[l1, l2, l3]*conj(B[l3, s2, c])
    @tensoropt L2[l1, s1, b, c] := L1[l1, l2, s2, c]*M[l2, s1, s2, b]
    @tensoropt new_L[a, b, c] := L2[l1, s1, b, c]*A[l1, s1, a]
    return new_L
end

"""
    prop_right4(L::Array{T, 4}, A::Array{T, 3}, M1::Array{T, 4},
                M2::Array{T, 4}, B::Array{T, 3}) where T<:Number

Propagate the tensor L through the tensors A, M1, M2, conj(B).

 ---- A ---         ---
 |    |             |
 |--- M1---         |--
 L    |      =  new_L
 |----M2---         |--
 |    |             |
 ---- B ---         ---

"""
function prop_right4(L::Array{T, 4}, A::Array{T, 3}, M1::Array{T, 4},
                     M2::Array{T, 4}, B::Array{T, 3}) where T<:Number
    @tensoropt L1[l1, l2, l3, s3, d] := L[l1, l2, l3, l4]*conj(B[l4, s3, d])
    @tensoropt L2[l1, l2, s2, c, d] := L1[l1, l2, l3, s3, d]*M2[l3, s2, s3, c]
    @tensoropt L3[l1, s1, b, c, d] := L2[l1, l2, s2, c, d]*M1[l2, s1, s2, b]
    @tensoropt new_L[a, b, c, d] := L3[l1, s1, b, c, d]*A[l1, s1, a]
    return new_L
end

"""
    prop_left2(A::Array{T, 3}, B::Array{T, 3}, R::Array{T, 2}) where T<:Number

Propagate the tensor R through the tensors A, conj(B).


 --- A ----      ---
     |    |        |
     |    R   =   R2
     |    |        |
 --- B ----      ---

"""
function prop_left2(A::Array{T, 3}, B::Array{T, 3},
                    R::Array{T, 2}) where T<:Number
    Ar = reshape(A, size(A, 1)*size(A, 2), size(A, 3))
    T1 = BLAS.gemm('N', 'N', Ar, R)
    T1 = reshape(T1, size(A, 1), size(A, 2)*size(T1, 2))
    Br = reshape(B, size(B, 1), size(B, 2)*size(B, 3))
    L2 = BLAS.gemm('N', 'C', T1, Br)
    return L2
end

"""
    prop_left3(A::Array{T, 3}, M::Array{T, 4}, B::Array{T, 3},
               R::Array{T, 3}) where T<:Number

Propagate the tensor R through the tensors A, M, conj(B).

 --- A ----      ---
     |    |        |
 --- M -- R   =    new_R
     |    |        |
 --- B ----      ---

"""
function prop_left3(A::Array{T, 3}, M::Array{T, 4}, B::Array{T, 3},
                    R::Array{T, 3}) where T<:Number
    @tensoropt new_R[a, b, c] := (
        A[a, s1, r1]*M[b, s1, s2, r2]*conj(B[c, s2, r3])*R[r1, r2, r3]
    )
    return new_R
end

"""
    prop_left4(A::Array{T, 3}, M1::Array{T, 4}, M2::Array{T, 4},
               B::Array{T, 3}, R::Array{T, 4}) where T<:Number

Propagate the tensor R through the tensors A, M1, M2, conj(B).

 --- A -----     ----
     |     |        |
 --- M1 ---|     ---|
     |     R   =    new_R
 --- M2 ---|     ---|
     |     |        |
 --- B -----     ----

"""
function prop_left4(A::Array{T, 3}, M1::Array{T, 4}, M2::Array{T, 4},
                    B::Array{T, 3}, R::Array{T, 4}) where T<:Number
    @tensoropt R1[a, s1, r2, r3, r4] := A[a, s1, r1]*R[r1, r2, r3, r4]
    @tensoropt R2[a, b, s2, r3, r4] := R1[a, s1, r2, r3, r4]*M1[b, s1, s2, r2]
    @tensoropt R3[a, b, c, s3, r4] := R2[a, b, s2, r3, r4]*M2[c, s2, s3, r3]
    @tensoropt new_R[a, b, c, d] := R3[a, b, c, s3, r4]*conj(B[d, s3, r4])
    return new_R
end

"""
    prop_right_subexp(L::Array{T, 3}, W::Array{T, 4}, M::Array{T, 3}) where T<:Number

Propagate the tensor L through the tensors W, conj(M) for the DMRG3S algorithm.

 ----               ---
 |                  |
 |    |             |----
 L -- W ---     =   P
 |    |             |----
 |    |             |
 ---- M ---         ---

"""
function prop_right_subexp(L::Array{T, 3}, W::Array{T, 4},
                           M::Array{T, 3}) where T<:Number
    @tensoropt P[l1, s1, r2, r3] := (
        L[l1, l2, l3]*conj(M[l3, s2, r3])*W[l2, s1, s2, r2]
        )
    return P
end

"""
    prop_left_subexp(W::Array{T, 4}, M::Array{T, 3},
                     R::Array{T, 3}) where T<:Number

Propagate the tensor R through the tensors W, conj(M) for the DMRG3S algorithm.

        ----     ----
           |        |
           |     ---|
     |     |   =    P
 --- W --- R     ---|
     |     |        |
     |     |        |
 --- B -----     ----

"""
function prop_left_subexp(W::Array{T, 4}, M::Array{T, 3},
                          R::Array{T, 3}) where T<:Number
    @tensoropt R1[r1, r2, s2, l3] := R[r1, r2, r3]*conj(M[l3, s2, r3])
    @tensoropt P[s1, r1, l2, l3] := R1[r1, r2, s2, l3]*W[l2, s1, s2, r2]
    return P
end

"""
    absorb_Le(Le::Array{T, 3}, M::Array{T, 2}) where T<:Number

Absorb M and conj(M) into the left environment Le.

 ----- M ------     ----
 |                  |
Le ------------ =  Le --
 |                  |
 --- conj(M) --     ----

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

Absorb M and conj(M) into the right environment Re.

 ---- M ------     ----
             |        |
 ---------- Re  =  -- Re
             |        |
 - conj(M) ---     ----

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

 ---      ----
 |     |     |     |
Le --- W --- Re =  Hi
 |     |     |     |
 ---      ----

"""
function build_local_hamiltonian(Le::Array{T, 3}, W::Array{T, 4},
                                 Re::Array{T, 3})  where T<:Number
    @tensoropt Hi[l1, s1, r1, l3, s2, r3] := (
        Le[l1, l2, l3]*W[l2, s1, s2, r2]*Re[r1, r2, r3]
        )
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
function build_local_hamiltonian(
    Le::Array{T, 3}, W::Array{T, 4}, Re::Array{T, 3}, cache::Cache{T}
    )  where T<:Number
    # Check if any cache element of appropriate size can be used.
    needed_space = (size(Le, 1)*size(W, 2)*size(Re, 1)
                    *size(Le, 3)*size(W, 3)*size(Re, 3))
    loc_in_cache = is_in_cache(cache, needed_space)
    if loc_in_cache > 0
        Hi = reshape(cache.elts[loc_in_cache],
                     (size(Le, 1), size(W, 2), size(Re, 1),
                      size(Le, 3), size(W, 3), size(Re, 3)))
        @tensoropt Hi[l1, s1, r1, l3, s2, r3] = (
            Le[l1, l2, l3]*W[l2, s1, s2, r2]*Re[r1, r2, r3]
            )
    else
        @tensoropt Hi[l1, s1, r1, l3, s2, r3] := (
            Le[l1, l2, l3]*W[l2, s1, s2, r2]*Re[r1, r2, r3]
            )
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

Build the local Hamiltonian with the left and right environments for DMRG2.

 ---            ---
 |    |     |     |    |
Le -- W1 -- W2 -- Re = Hi
 |    |     |     |    |
 ---            ---

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
        @tensoropt Hi[l1, s1, s3, r1, l3, s2, s4, r3] = (
            Le[l1, l2, l3]*W1[l2, s1, s2, a]*W2[a, s3, s4, r2]*Re[r1, r2, r3]
            )
    else
        @tensoropt Hi[l1, s1, s3, r1, l3, s2, s4, r3] := (
            Le[l1, l2, l3]*W1[l2, s1, s2, a]*W2[a, s3, s4, r2]*Re[r1, r2, r3]
            )
    end
    Hi = reshape(Hi, ((size(Le, 1)*size(W1, 2)*size(W2, 2)*size(Re, 1),
                       size(Le, 3)*size(W1, 3)*size(W2, 3)*size(Re, 3))))
    update_cache!(cache, Hi)
    return Hi
end
