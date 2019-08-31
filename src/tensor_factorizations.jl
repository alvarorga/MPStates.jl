export factorize_qr,
       factorize_lq

"""
    factorize_qr(M::Array{T, 3}) where T<:Number

QR factorization of M tensor.
"""
function factorize_qr(M::Array{T, 3}) where T<:Number
    m1, m2, m3 = size(M)
    M = reshape(M, m1*m2, m3)
    Q, R = qr(M)
    Q = Matrix(Q)
    Q = reshape(Q, m1, m2, size(Q, 2))
    return Q, R
end

"""
    factorize_lq(M::Array{T, 3}) where T<:Number

LQ factorization of M tensor.
"""
function factorize_lq(M::Array{T, 3}) where T<:Number
    m1, m2, m3 = size(M)
    M = reshape(M, m1, m2*m3)
    L, Q = lq(M)
    Q = Matrix(Q)
    Q = reshape(Q, size(Q, 1), m2, m3)
    return L, Q
end
