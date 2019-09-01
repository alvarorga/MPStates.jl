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
                        normalize_S::Bool)

Do SVD with cutoff truncation and normalization of singular values.
"""
function do_svd_with_options(M::AbstractMatrix{<:Number},
                             cutoff::Float64,
                             normalize_S::Bool)
    F = svd(M)
    S = F.S
    ix_cutoff = findlast(S .> cutoff)

    S = Diagonal(S[1:ix_cutoff])
    if normalize_S
        S ./= norm(S)
    end
    U = F.U[:, 1:ix_cutoff]
    Vt = F.Vt[1:ix_cutoff, :]
    return U, S, Vt
end

"""
    factorize_svd_right(M::Array{<:Number, 3};
                        cutoff::Float64=1e-8,
                        normalize_S::Bool=true)

SVD factorization of M tensor reshapen to (M[1]*M[2], M[3]*1).
"""
function factorize_svd_right(M::Array{<:Number, 3};
                             cutoff::Float64=1e-8,
                             normalize_S::Bool=true)
    U, SVt = factorize_svd_right(reshape(M, size(M)..., 1), cutoff, normalize_S)
    SVt = reshape(SVt, size(SVt, 1), size(SVt, 2))
    return U, SVt
end

"""
    factorize_svd_left(M::Array{<:Number, 3};
                       cutoff::Float64=1e-8,
                       normalize_S::Bool=true)

SVD factorization of M tensor reshapen to (1*M[1], M[2]*M[3]).
"""
function factorize_svd_left(M::Array{<:Number, 3};
                            cutoff::Float64=1e-8,
                            normalize_S::Bool=true)
    US, Vt = factorize_svd_left(reshape(M, 1, size(M)...), cutoff, normalize_S)
    US = reshape(US, size(US, 2), size(US, 3))
    return US, Vt
end

"""
    factorize_svd_right(M::Array{<:Number, 4};
                        cutoff::Float64=1e-8,
                        normalize_S::Bool=true)

SVD factorization of M tensor reshapen to (M[1]*M[2], M[3]*M[4]).
"""
function factorize_svd_right(M::Array{<:Number, 4};
                             cutoff::Float64=1e-8,
                             normalize_S::Bool=true)
    m1, m2, m3, m4 = size(M)
    M = reshape(M, m1*m2, m3*m4)
    U, S, Vt = do_svd_with_options(M, cutoff, normalize_S)
    SVt = S*Vt
    U = reshape(U, m1, m2, size(U, 2))
    SVt = reshape(SVt, size(S, 1), m3, m4)
    return U, SVt
end

"""
    factorize_svd_right(M::Array{<:Number, 4}
                        cutoff::Float64=1e-8,
                        normalize_S::Bool=true)

SVD factorization of M tensor reshapen to (M[1]*M[2], M[3]*M[4]).
"""
function factorize_svd_left(M::Array{<:Number, 4};
                            cutoff::Float64=1e-8,
                            normalize_S::Bool=true)
    m1, m2, m3, m4 = size(M)
    M = reshape(M, m1*m2, m3*m4)
    U, S, Vt = do_svd_with_options(M, cutoff, normalize_S)
    US = U*S
    US = reshape(US, m1, m2, size(US, 2))
    Vt = reshape(Vt, size(Vt, 1), m3, m4)
    return US, Vt
end
