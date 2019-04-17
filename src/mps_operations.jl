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
    contract(psi::Mps, phi::Mps)

Contraction of two MPS: <psi|phi>.
"""
function contract(psi::Mps{T}, phi::Mps{T}) where T
    L = Matrix{T}(I, 1, 1)
    for i=1:psi.L
        @tensor L[c, e] := conj(psi.A[i][b, d, e])*L[a, b]*phi.A[i][a, d, c]
    end
    return L[1, 1]
end
