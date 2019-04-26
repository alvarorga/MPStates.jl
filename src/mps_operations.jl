#
# Operations between MPSs.
#

"""
    m_occupation(psi::Mps{T}, i::Int, s::Int=2) where T<:Number

Measure occupation at site `i` of the local population in state `s`.
For example, if the physical dimension of `psi` is 2, then measuring
at `s=2` is the same as measuring the number of particles, while at
`s=1` measures the number of holes.
"""
function m_occupation(psi::Mps{T}, i::Int, s::Int=2) where T<:Number
    Ai = psi.A[i][:, s, :]
    L = transpose(Ai)*conj(Ai)
    for j=i+1:psi.L
        L = prop_right2(L, psi.A[j], psi.A[j])
    end
    return L[1, 1]
end

"""
    m_fermionic_correlation(psi::Mps{T}, i::Int, j::Int) where T<:Number

Measure correlation <c^dagger_i c_j>, with `psi` a fermionic state.
"""
function m_fermionic_correlation(psi::Mps{T}, i::Int, j::Int) where T<:Number
    psi.d == 2 || throw("Physical dimension of Mps is not 2.")
    i != j || throw("Site i must be different than j.")

    # Operators c^dagger, c, and (1-2n).
    cdag = zeros(T, 1, 2, 2, 1)
    cdag[1, 2, 1, 1] = one(T)
    c = zeros(T, 1, 2, 2, 1)
    c[1, 1, 2, 1] = one(T)
    Z = zeros(T, 1, 2, 2, 1)
    Z[1, 1, 1, 1] = one(T)
    Z[1, 2, 2, 1] = -one(T)

    L = ones(T, 1, 1, 1)
    if i < j
        L = prop_right3(L, psi.A[i], cdag, psi.A[i])
        for k=i+1:j-1
            L = prop_right3(L, psi.A[k], Z, psi.A[k])
        end
        L = prop_right3(L, psi.A[j], c, psi.A[j])
    else
        L = prop_right3(L, psi.A[j], c, psi.A[j])
        for k=i+1:j-1
            L = prop_right3(L, psi.A[k], Z, psi.A[k])
        end
        L = prop_right3(L, psi.A[i], cdad, psi.A[i])
    end
    return L[1, 1, 1]
end

"""
    m_correlation(psi::Mps{T}, i::Int, j::Int) where T<:Number

Measure correlation <c^dagger_i c_j>, with `psi` a non fermionic state.
"""
function m_correlation(psi::Mps{T}, i::Int, j::Int) where T<:Number
    psi.d == 2 || throw("Physical dimension of Mps is not 2.")
    i != j || throw("Site i must be different than j.")

    # Operators c^dagger, c, and (1-2n).
    cdag = zeros(T, 1, 2, 2, 1)
    cdag[1, 2, 1, 1] = one(T)
    c = zeros(T, 1, 2, 2, 1)
    c[1, 1, 2, 1] = one(T)
    Id = zeros(T, 1, 2, 2, 1)
    Id[1, 1, 1, 1] = one(T)
    Id[1, 2, 2, 1] = one(T)

    L = ones(T, 1, 1, 1)
    if i < j
        L = prop_right3(L, psi.A[i], cdag, psi.A[i])
        for k=i+1:j-1
            L = prop_right3(L, psi.A[k], Id, psi.A[k])
        end
        L = prop_right3(L, psi.A[j], c, psi.A[j])
    else
        L = prop_right3(L, psi.A[j], c, psi.A[j])
        for k=i+1:j-1
            L = prop_right3(L, psi.A[k], Id, psi.A[k])
        end
        L = prop_right3(L, psi.A[i], cdad, psi.A[i])
    end
    return L[1, 1, 1]
end

"""
    contract(psi::Mps, phi::Mps) where T

Contraction of two MPS: <psi|phi>.
"""
function contract(psi::Mps{T}, phi::Mps{T}) where T
    L = Matrix{T}(I, 1, 1)
    for i=1:psi.L
        L = prop_right2(L, phi.A[i], psi.A[i])
    end
    return L[1, 1]
end

"""
    schmidt_decomp(psi::Mps{T}, i::Int) where T

Compute the singular values of the Schmidt decomposition of state `psi` between
sites `i` and `i+1`.
"""
function schmidt_decomp(psi::Mps{T}, i::Int) where T
    # The tensors 1 to i are already left-canonical in A, but we must write
    # the A tensors from i+1 to end in right-canonical form. When we do this
    # we won't need the information contained in the Q tensors.
    L = Matrix{T}(I, 1, 1)
    for j=psi.L:-1:i+1
        L = prop_lq(psi.A[j], L, false)
    end
    return svdvals!(L)
end

"""
    ent_entropy(psi::Mps{T}, i::Int) where T

Compute the entanglement entropy of state `psi` between sites `i` and `i+1`.
"""
function ent_entropy(psi::Mps{T}, i::Int) where T
    rho = schmidt_decomp(psi, i)
    S = 0.
    for j=1:length(rho)
        abs(rho[j]) < 1e-10 && continue
        S -= rho[j]*log2(rho[j])
    end
    return S
end

"""
    enlarge_bond_dimension(psi::Mps{T}, max_D::Int) where T

Take a state and add 0's to the tensors until a maximum bond dimension is
reached.
"""
function enlarge_bond_dimension!(psi::Mps{T}, max_D::Int) where T
    L = psi.L
    # Resize A.
    for i=1:L
        d1 = size(psi.A[i], 1)
        d2 = size(psi.A[i], 3)
        # Check if d1 needs to be resized.
        need_resize_d1 = log2(d1) < minimum([i-1, L-i+1, log2(max_D)])
        if log2(d1) < log2(max_D) < minimum([i-1, L-i+1])
            new_d1 = max_D
        elseif log2(d1) < minimum([i-1, L-i+1]) < log2(max_D)
            new_d1 = 1<<minimum([i-1, L-i+1])
        else
            new_d1 = d1
        end
        # Check if d2 needs to be resized.
        need_resize_d2 = log2(d2) < minimum([i, L-i, log2(max_D)])
        if log2(d2) < log2(max_D) < minimum([i, L-i])
            new_d2 = max_D
        elseif log2(d1) < minimum([i, L-i]) < log2(max_D)
            new_d2 = 1<<minimum([i, L-i])
        else
            new_d2 = d2
        end

        # Resize A with the appropriate dimensions, if needed.
        if need_resize_d1 || need_resize_d2
            new_A = zeros(T, new_d1, psi.d, new_d2)
            new_A[1:d1, :, 1:d2] = psi.A[i]
            psi.A[i] = new_A
        end
    end

    # Resize B.
    for i=1:L
        d1 = size(psi.B[i], 1)
        d2 = size(psi.B[i], 3)
        need_resize_d1 = log2(d1) < minimum([i-1, L-i+1, log2(max_D)])
        if log2(d1) < log2(max_D) < minimum([i-1, L-i+1])
            new_d1 = max_D
        elseif log2(d1) < minimum([i-1, L-i+1]) < log2(max_D)
            new_d1 = 1<<minimum([i-1, L-i+1])
        else
            new_d1 = d1
        end
        # Check if d2 needs to be resized.
        need_resize_d2 = log2(d2) < minimum([i, L-i, log2(max_D)])
        if log2(d2) < log2(max_D) < minimum([i, L-i])
            new_d2 = max_D
        elseif log2(d1) < minimum([i, L-i]) < log2(max_D)
            new_d2 = 1<<minimum([i, L-i])
        else
            new_d2 = d2
        end

        # Resize B with the appropriate dimensions, if needed.
        if need_resize_d1 || need_resize_d2
            new_B = zeros(T, new_d1, psi.d, new_d2)
            new_B[1:d1, :, 1:d2] = psi.B[i]
            psi.B[i] = new_B
        end
    end
    return psi
end
