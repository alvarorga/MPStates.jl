export factorize_qr,
       factorize_lq,
       factorize_svd_right,
       factorize_svd_left

"""
    factorize_qr(M::Array{<:Number, 3})

QR factorization of M tensor.
"""
function factorize_qr(M::Array{<:Number, 3})
    m1, m2, m3 = size(M)
    M = reshape(M, m1*m2, m3)
    Q, R = qr(M)
    Q = Matrix(Q)
    Q = reshape(Q, m1, m2, size(Q, 2))
    return Q, R
end

"""
    factorize_lq(M::Array{<:Number, 3})

LQ factorization of M tensor.
"""
function factorize_lq(M::Array{<:Number, 3})
    m1, m2, m3 = size(M)
    M = reshape(M, m1, m2*m3)
    L, Q = lq(M)
    Q = Matrix(Q)
    Q = reshape(Q, size(Q, 1), m2, m3)
    return L, Q
end

"""
    do_svd_with_options(M::AbstractMatrix{<:Number},
                        cutoff::Float64,
                        dimcutoff::Int,
                        normalize_S::Bool)

Do SVD with cutoff truncation and normalization of singular values.
"""
function do_svd_with_options(M::AbstractMatrix{<:Number};
                             cutoff::Float64=1e-8,
                             dimcutoff::Int=1000,
                             normalize_S::Bool=true)
    F = svd(M)
    S = F.S
    ix_cutoff = min(findlast(S .> cutoff), dimcutoff)

    S = Diagonal(S[1:ix_cutoff])
    if normalize_S
        S ./= norm(S)
    end
    U = F.U[:, 1:ix_cutoff]
    Vt = F.Vt[1:ix_cutoff, :]
    return U, S, Vt
end

"""
    factorize_svd_right(M::AbstractMatrix{<:Number}; kwargs...)

SVD factorization of M matrix (U unitary).

 -- M -- = -- U -- SVt --
"""
function factorize_svd_right(M::AbstractMatrix{<:Number}; kwargs...)
    m1, m2 = size(M)
    U, S, Vt = do_svd_with_options(M; kwargs...)
    SVt = S*Vt
    return U, SVt
end

"""
    factorize_svd_left(M::AbstractMatrix{<:Number}; kwargs...)

SVD factorization of M matrix (Vt unitary).

 -- M -- = -- US -- Vt --
"""
function factorize_svd_left(M::AbstractMatrix{<:Number}; kwargs...)
    m1, m2 = size(M)
    U, S, Vt = do_svd_with_options(M; kwargs...)
    US = U*S
    return US, Vt
end

"""
    factorize_svd_right(M::Array{<:Number, 3}; kwargs...)

SVD factorization of M tensor (U unitary).

 -- M -- = -- U -- SVt --
    |         |
"""
function factorize_svd_right(M::Array{<:Number, 3}; kwargs...)
    m1, m2, m3 = size(M)
    U, SVt = factorize_svd_right(reshape(M, m1*m2, m3); kwargs...)
    U = reshape(U, m1, m2, size(U, 2))
    return U, SVt
end

"""
    factorize_svd_left(M::Array{<:Number, 3}; kwargs...)

SVD factorization of M tensor (Vt unitary).

 -- M -- = -- US -- Vt --
    |               |
"""
function factorize_svd_left(M::Array{<:Number, 3}; kwargs...)
    m1, m2, m3 = size(M)
    US, Vt = factorize_svd_left(reshape(M, m1, m2*m3); kwargs...)
    Vt = reshape(Vt, size(Vt, 1), m2, m3)
    return US, Vt
end

"""
    factorize_svd_right(M::Array{<:Number, 4}; kwargs...)

SVD factorization of M tensor (U unitary).

 -- M -- = -- U -- SVt --
   | |        |     |
"""
function factorize_svd_right(M::Array{<:Number, 4}; kwargs...)
    m1, m2, m3, m4 = size(M)
    U, SVt = factorize_svd_right(reshape(M, m1*m2, m3*m4); kwargs...)
    U = reshape(U, m1, m2, size(U, 2))
    SVt = reshape(SVt, size(SVt, 1), m3, m4)
    return U, SVt
end

"""
    factorize_svd_left(M::Array{<:Number, 4}; kwargs...)

SVD factorization of M tensor (Vt unitary).

 -- M -- = -- US -- Vt --
   | |        |     |
"""
function factorize_svd_left(M::Array{<:Number, 4}; kwargs...)
    m1, m2, m3, m4 = size(M)
    US, Vt = factorize_svd_left(reshape(M, m1*m2, m3*m4); kwargs...)
    US = reshape(US, m1, m2, size(US, 2))
    Vt = reshape(Vt, size(Vt, 1), m3, m4)
    return US, Vt
end
