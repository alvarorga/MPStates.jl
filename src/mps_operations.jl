#
# Operations between MPSs.
#

"""
    expected(psi::Mps{T}, op_i::AbstractMatrix{<:Number}, i::Int) where T<:Number

Compute expected value of the operator op_i in the state psi.
"""
function expected(psi::Mps{T}, op_i::AbstractMatrix{<:Number}, i::Int) where T<:Number
    # Convert op_i to have the same type as psi and reshape so that it can pass
    # in prop_right3.
    c_op_i = convert.(T, reshape(op_i, 1, size(op_i, 1), size(op_i, 2), 1))

    L = ones(T, 1, 1)
    for s=1:psi.L
        if s != i
            L = prop_right2(L, psi.M[s], psi.M[s])
        else
            # Reshape L to a rank 3 tensor so that it can pass in prop_right3
            # and then reshape it back to rank 2.
            L3 = reshape(L, size(L, 1), 1, size(L, 2))
            L3 = prop_right3(L3, psi.M[s], c_op_i, psi.M[s])
            L = reshape(L3, size(L3, 1), size(L3, 3))
        end
    end
    return L[1, 1]
end

"""
    expected(psi::Mps{T}, op_i::AbstractMatrix{<:Number}, i::Int) where T<:Number

Compute expected value of the operator (str_)op_i in the state psi.
"""
function expected(psi::Mps{T}, str_op_i::String, i::Int) where T<:Number
    return expected(psi, str_to_op(str_op_i), i)
end

"""
    expected(psi::Mps{T}, op_i::AbstractMatrix{<:Number}, i::Int,
             op_j::AbstractMatrix{<:Number}, j::Int;
             ferm_op::AbstractMatrix{<:Number}=Matrix{T}(I, psi.d, psi.d)
             ) where T<:Number

Compute expected value of the operator op_i*op_j in the state psi. An optional
fermionic parity operator can be passed in ferm_op.
"""
function expected(psi::Mps{T}, op_i::AbstractMatrix{<:Number}, i::Int,
                  op_j::AbstractMatrix{<:Number}, j::Int;
                  ferm_op::AbstractMatrix{<:Number}=Matrix{T}(I, psi.d, psi.d)
                  ) where T<:Number
    # Convert op_i, op_j, and ferm_op to have the same type as psi and reshape
    # to a rank 4 tensor so that they can pass in prop_right3.
    c_op_i = convert.(T, reshape(op_i, 1, size(op_i, 1), size(op_i, 2), 1))
    c_op_j = convert.(T, reshape(op_j, 1, size(op_j, 1), size(op_j, 2), 1))
    c_ferm_op = convert.(T, reshape(ferm_op, 1, size(ferm_op, 1),
                                    size(ferm_op, 2), 1))

    # Sort operators.
    s1 = min(i, j)
    op_1 = i < j ? c_op_i : c_op_j
    s2 = max(i, j)
    op_2 = i > j ? c_op_i : c_op_j

    L = ones(T, 1, 1)
    for s=1:s1-1
        L = prop_right2(L, psi.M[s], psi.M[s])
    end
    # Reshape L to a rank 3 tensor so that it can pass in prop_right3.
    L3 = reshape(L, size(L, 1), 1, size(L, 2))
    L3 = prop_right3(L3, psi.M[s1], op_1, psi.M[s1])
    # Between min(i, j) and max(i, j) apply the fermionic parity operator.
    for s=s1+1:s2-1
        L3 = prop_right3(L3, psi.M[s], c_ferm_op, psi.M[s])
    end
    L3 = prop_right3(L3, psi.M[s2], op_2, psi.M[s2])
    # Reshape L3 back to a rank 2 tensor.
    L = reshape(L3, size(L3, 1), size(L3, 3))
    for s=s2+1:psi.L
        L = prop_right2(L, psi.M[s], psi.M[s])
    end

    return L[1, 1]
end

"""
    expected(psi::Mps{T}, op_i::AbstractMatrix{<:Number}, i::Int,
             op_j::AbstractMatrix{<:Number}, j::Int;
             ferm_op::AbstractMatrix{<:Number}=Matrix{T}(I, psi.d, psi.d)
             ) where T<:Number

Compute expected value of the operator op_i*op_j in the state psi. An optional
fermionic parity operator can be passed in ferm_op.
"""
function expected(psi::Mps{T}, op_i::String, i::Int, op_j::String, j::Int;
                  ferm_op::String="none") where T<:Number
    if ferm_op != "none"
        return expected(psi, str_to_op(op_i), i, str_to_op(op_j), j,
                        ferm_op=str_to_op(ferm_op))
   else
        return expected(psi, str_to_op(op_i), i, str_to_op(op_j), j)
   end
end

"""
    contract(psi::Mps{T}, phi::Mps{T}) where T<:Number

Contraction of two MPS: <psi|phi>.
"""
function contract(psi::Mps{T}, phi::Mps{T}) where T<:Number
    L = Matrix{T}(I, 1, 1)
    for i=1:psi.L
        L = prop_right2(L, phi.M[i], psi.M[i])
    end
    return L[1, 1]
end

"""
    norm(psi::Mps{T}) where T<:Number

Norm of a MPS: <psi|psi>. Extend the LinearAlgebra module function.
"""
function LinearAlgebra.norm(psi::Mps{T}) where T<:Number
    return contract(psi, psi)
end

"""
    schmidt_decomp(psi::Mps{T}, i::Int) where T<:Number

Compute the singular values of the Schmidt decomposition of state `psi` between
sites `i` and `i+1`.
"""
function schmidt_decomp(psi::Mps{T}, i::Int) where T<:Number
    # Write the tensors 1 -> i in left canonical form and the tensors i+1 -> L
    # in right canonical form. Discard the information in of the unitary
    # matrices Q.
    # Left canonical form.
    L = ones(T, 1, 1)
    for j=1:i
        L = prop_qr(L, psi.M[j], false)
    end
    R = ones(T, 1, 1)
    for j=reverse(i+1:psi.L)
        R = prop_lq(psi.M[j], R, false)
    end
    return svdvals!(L*R)
end

"""
    ent_entropy(psi::Mps{T}, i::Int) where T<:Number

Compute the entanglement entropy of state `psi` between sites `i` and `i+1`.
"""
function ent_entropy(psi::Mps{T}, i::Int) where T<:Number
    rho = schmidt_decomp(psi, i)
    S = 0.
    for j=1:length(rho)
        abs(rho[j]) < 1e-10 && continue
        S -= rho[j]*log2(rho[j])
    end
    return S
end

"""
    bond_dimension_with_m(L::Int, i::Int, m::Int, d::Int)

Compute the appropriate bond dimension at site `i` for an Mps of length `L`,
physical dimension `d`, and maximum bond dimension `m`.

There are L+1 possible bonds: the bond at i=1 has dimension 1 and so does the
bond at i=L+1. The bond at site i=2 has dimension d, like the bond at i=L.
"""
function bond_dimension_with_m(L::Int, i::Int, m::Int, d::Int)
    mi = m
    if i <= L>>1
        if (i-1)*log(d) < log(m)
            mi = d^(i-1)
        end
    else
        if (L+1-i)*log(d) < log(m)
            mi = d^(L+1-i)
        end
    end
    return mi
end

"""
    enlarge_bond_dimension!(psi::Mps{T}, max_D::Int) where T<:Number

Take a state and add 0's to the tensors until a maximum bond dimension is
reached.
"""
function enlarge_bond_dimension!(psi::Mps{T}, max_D::Int) where T<:Number
    L = psi.L

    for i=1:L
        m1 = size(psi.M[i], 1)
        m2 = size(psi.M[i], 3)
        # Check if m1 and m2 need to be resized.
        new_m1 = bond_dimension_with_m(psi.L, i, max_D, psi.d)
        new_m2 = bond_dimension_with_m(psi.L, i+1, max_D, psi.d)
        need_resize_m1 = m1 < new_m1
        need_resize_m2 = m2 < new_m2
        # Resize M with the appropriate dimensions, if needed.
        if need_resize_m1 || need_resize_m2
            new_M = zeros(T, new_m1, psi.d, new_m2)
            new_M[1:m1, :, 1:m2] = psi.M[i]
            psi.M[i] = new_M
        end
    end
    return psi
end

"""
    svd_truncate!(psi::Mps{T}, max_D::Int) where T<:Number

Truncate the bond dimension of `psi` by `max_D` using SVD decomposition.
"""
function svd_truncate!(psi::Mps{T}, max_D::Int) where T<:Number
    US = ones(T, 1, 1)
    for i=reverse(2:psi.L)
        US, new_M = prop_left_svd(psi.M[i], US, max_D)
        psi.M[i] = new_M
    end
    # Contract and normalize last tensor.
    @tensor new_M[i, s, j] := psi.M[1][i, s, k]*US[k, j]
    psi.M[1] = new_M./norm(new_M)

    return psi
end

"""
    simplify!(psi::Mps{T}, D::Int;
              max_sweeps::Int=500, tol::Float64=1e-6) where T<:Number

Variationally simplify an Mps to make it have bond dimension `D`.
"""
function simplify!(psi::Mps{T}, D::Int;
                   max_sweeps::Int=500, tol::Float64=1e-6) where T<:Number
    # Normalize state if it is not normalized.
    norm_psi = norm(psi)
    psi.M[end] ./= sqrt(norm_psi)

    # Copy original state to make it the objective state to which we want to
    # approximate. Make it a new Mps.
    obj_M = deepcopy(psi.M)
    obj = Mps(obj_M, psi.L, psi.d)
    # Reduce the bond dimension of the state to the wanted one.
    svd_truncate!(psi, D)

    # Create left and right environments.
    Le = fill(ones(T, 1, 1), psi.L)
    Re = fill(ones(T, 1, 1), psi.L)
    # Initialize left environment.
    for i=2:psi.L
        Le[i] = prop_right2(Le[i-1], psi.M[i-1], obj.M[i-1])
    end

    # Compute distance of new state to the objective one.
    dist = [2. - 2*real(contract(psi, obj))]
    dist_converged = false
    nsweep = 2
    while dist[nsweep-1] > tol && nsweep <= max_sweeps+1 && !dist_converged
        # Do left and right sweeps.
        do_sweep_simplify!(psi, obj, Le, Re, -1)
        do_sweep_simplify!(psi, obj, Le, Re, +1)
        # Compute distance of `psi` to objective after sweeps. Use abs to
        # remove any phase that could enter in `psi`.
        push!(dist, 2. - 2*abs(real(contract(psi, obj))))
        # Update while loop control parameters.
        dist_converged = abs(dist[nsweep] - dist[nsweep-1]) < 1e-8
        nsweep += 1
    end

    return psi
end

"""
    do_sweep_simplify!(psi::Mps{T}, obj::Mps{T},
                       Le::Vector{Array{T, 2}}, Re::Vector{Array{T, 2}},
                       sense::Int) where T<:Number

Do a sweep to variationally simplify `psi`.
"""
function do_sweep_simplify!(psi::Mps{T}, obj::Mps{T},
                            Le::Vector{Array{T, 2}}, Re::Vector{Array{T, 2}},
                            sense::Int) where T<:Number

    sense == 1 || sense == -1 || throw("`Sense` must be either `+1` or `-1`.")
    # Order of sites to do the sweep.
    sweep_sites = sense == +1 ? (1:psi.L) : reverse(1:psi.L)
    for i in sweep_sites
        # Compute local tensor.
        @tensor Mi[l1, s, r1] := Le[i][l1, l2]*obj.M[i][l2, s, r2]*Re[i][r1, r2]
        Mi = conj(Mi)

        # Update left and right environments.
        if sense == +1
            Mi = reshape(Mi, (size(Le[i], 1)*psi.d, size(Re[i], 1)))
            Qa, Ra = qr(Mi)
            Qa = Matrix(Qa)
            Qa = reshape(Qa, (size(Le[i], 1), psi.d, size(Qa, 2)))

            # Update left and right environments at L[i+1] and R[i] and psi.
            psi.M[i] = Qa
            if i < psi.L
                Le[i+1] = prop_right2(Le[i], Qa, obj.M[i])
                Re[i] = Ra*Re[i]
            end
        else
            Mi = reshape(Mi, (size(Le[i], 1), psi.d*size(Re[i], 1)))
            La, Qa = lq(Mi)
            Qa = Matrix(Qa)
            Qa = reshape(Qa, (size(Qa, 1), psi.d, size(Re[i], 1)))

            # Update left and right environments at L[i] and R[i-1] and psi.
            psi.M[i] = Qa
            if i > 1
                Re[i-1] = prop_left2(Qa, obj.M[i], Re[i])
                Le[i] = transpose(La)*Le[i]
            end
        end
    end
    return psi
end

#
# INPUT/OUTPUT OF MPS.
#

"""
    save_mps(filename::String, psi::Mps{T}) where T<:Number

Save the state `psi` in file `filename` in HDF5 format:
- `L`: length of Mps.
- `d`: physical dimension.
- `T`: type of Mps tensor elements.
- `Ai`: left canonical tensors.

More info about the HDF5 format can be found here:
    https://github.com/JuliaIO/HDF5.jl/blob/master/doc/hdf5.md
"""
function save_mps(filename::String, psi::Mps{T}) where T<:Number
    h5write(filename, "L", psi.L)
    h5write(filename, "d", psi.d)
    h5write(filename, "T", "$T")
    is_complex = !(T <: Real)
    for i=1:psi.L
        if !is_complex
            h5write(filename, "M$i", psi.M[i])
        else
            h5write(filename, "realM_$i", real(psi.M[i]))
            h5write(filename, "imagM_$i", imag(psi.M[i]))
        end
    end
    return
end

"""
    read_mps(filename::String)

Read the state `psi` in stored in file `filename` inHDF5 format.
"""
function read_mps(filename::String)
    L = h5read(filename, "L")
    d = h5read(filename, "d")
    str_T = h5read(filename, "T")
    if str_T == "Float32"
        T = Float32
    elseif str_T == "Float64"
        T = Float64
    elseif str_T == "Complex{Float32}"
        T = ComplexF32
    elseif str_T == "Complex{Float64}"
        T = ComplexF64
    end
    is_complex = !(T <: Real)

    # Start Mps as full and then replace tensors.
    psi = init_mps(T, L, "full")
    for i=1:psi.L
        if !is_complex
            psi.M[i] = h5read(filename, "M$i")
        else
            psi.M[i] = complex.(h5read(filename, "realM_$i"),
                                h5read(filename, "imagM_$i"))
        end
    end

    return psi
end
