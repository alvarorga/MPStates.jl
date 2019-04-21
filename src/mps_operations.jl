#
# Operations between MPSs.
#

"""
    m_occupation(psi::Mps, i::Int, s::Int=2)

Measure occupation at site `i` of the local population in state `s`.
For example, if the physical dimension of `psi` is 2, then measuring
at `s=2` is the same as measuring the number of particles, while at
`s=1` measures the number of holes.
"""
function m_occupation(psi::Mps, i::Int, s::Int=2)
    Ai = psi.A[i][:, s, :]
    L = transpose(Ai)*conj(Ai)
    for j=i+1:psi.L
        Aj = psi.A[j]
        @tensor L[a, b] := Aj[c, e, a]*L[c, d]*conj(Aj[d, e, b])
    end
    return L[1, 1]
end

"""
    contract(psi::Mps, phi::Mps) where T

Contraction of two MPS: <psi|phi>.
"""
function contract(psi::Mps{T}, phi::Mps{T}) where T
    L = Matrix{T}(I, 1, 1)
    for i=1:psi.L
        @tensor L[c, e] := conj(psi.A[i][b, d, e])*L[a, b]*phi.A[i][a, d, c]
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
    L = Matrix{T}(I, size(psi.A[end], 3), size(psi.A[end], 3))
    for j=psi.L:-1:i+1
        @tensor lA[a, s, c] := psi.A[j][a, s, b]*L[b, c]
        rA = reshape(lA, (size(lA, 1), size(lA, 2)*size(lA, 3)))
        L, Q = lq(rA)
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
