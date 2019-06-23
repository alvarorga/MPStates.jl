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
    return real(L[1, 1])
end

"""
    m_fermionic_correlation(psi::Mps{T}, i::Int, j::Int) where T<:Number

Measure the correlation <c^dagger_i c_j>, with `psi` a fermionic state.
"""
function m_fermionic_correlation(psi::Mps{T}, i::Int, j::Int) where T<:Number
    return m_generic_correlation(psi, i, j, true)
end

"""
    m_correlation(psi::Mps{T}, i::Int, j::Int) where T<:Number

Measure correlation <c^dagger_i c_j>, with `psi` a non fermionic state.
"""
function m_correlation(psi::Mps{T}, i::Int, j::Int) where T<:Number
    return m_generic_correlation(psi, i, j, false)
end

"""
    m_generic_correlation(psi::Mps{T}, i::Int, j::Int, is_fermionic::Bool) where T<:Number

Measure the correlation <c^dagger_i c_j>, with `psi` a fermion or boson state.
"""
function m_generic_correlation(psi::Mps{T}, i::Int, j::Int, is_fermionic::Bool) where T<:Number
    psi.d == 2 || throw("Physical dimension of Mps is not 2.")
    i != j || throw("Site i must be different than j.")

    # Operators c^dagger_i, c_j, Id, and (1-2n).
    cdi = zeros(T, 1, 2, 2, 1)
    cdi[1, 1, 2, 1] = 1.
    cj = zeros(T, 1, 2, 2, 1)
    cj[1, 2, 1, 1] = 1.
    Z = zeros(T, 1, 2, 2, 1)
    Z[1, 1, 1, 1] = 1.
    Z[1, 2, 2, 1] = -1.
    Id = zeros(T, 1, 2, 2, 1)
    Id[1, :, :, 1] = Matrix{T}(I, 2, 2)

    L = ones(T, 1, 1, 1)
    for k=1:psi.L
        if k == i
            L = prop_right3(L, psi.A[k], cdi, psi.A[k])
        elseif k == j
            L = prop_right3(L, psi.A[k], cj, psi.A[k])
        elseif is_fermionic && (i < k < j || j < k < i)
            L = prop_right3(L, psi.A[k], Z, psi.A[k])
        else
            L = prop_right3(L, psi.A[k], Id, psi.A[k])
        end
    end
    return L[1, 1, 1]
end

"""
    m_2occupations(psi::Mps{T}, i::Int, j::Int) where T<:Number

Measure the two point occupation <n_i n_j>.
"""
function m_2occupations(psi::Mps{T}, i::Int, j::Int) where T<:Number
    psi.d == 2 || throw("Physical dimension of Mps is not 2.")
    i != j || throw("Site i must be different than j.")

    # Operators n, and Id.
    n = zeros(T, 1, 2, 2, 1)
    n[1, 2, 2, 1] = 1.
    Id = zeros(T, 1, 2, 2, 1)
    Id[1, :, :, 1] = Matrix{T}(I, 2, 2)

    # Make tensor contraction.
    L = ones(T, 1, 1, 1)
    for k=1:psi.L
        if k == i || k == j
            L = prop_right3(L, psi.A[k], n, psi.A[k])
        else
            L = prop_right3(L, psi.A[k], Id, psi.A[k])
        end
    end
    return real(L[1, 1, 1])
end

"""
    contract(psi::Mps{T}, phi::Mps{T}) where T<:Number

Contraction of two MPS: <psi|phi>.
"""
function contract(psi::Mps{T}, phi::Mps{T}) where T<:Number
    L = Matrix{T}(I, 1, 1)
    for i=1:psi.L
        L = prop_right2(L, phi.A[i], psi.A[i])
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
    enlarge_bond_dimension!(psi::Mps{T}, max_D::Int) where T<:Number

Take a state and add 0's to the tensors until a maximum bond dimension is
reached.
"""
function enlarge_bond_dimension!(psi::Mps{T}, max_D::Int) where T<:Number
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
    return psi
end

"""
    svd_truncate!(psi::Mps{T}, max_D::Int) where T<:Number

Truncate the bond dimension of `psi` by `max_D` using SVD decomposition.
"""
function svd_truncate!(psi::Mps{T}, max_D::Int) where T<:Number
    # Truncate A.
    US = ones(T, 1, 1)
    for i=psi.L:-1:2
        US, new_A = prop_left_svd(psi.A[i], US, max_D)
        psi.A[i] = new_A
    end
    # Contract and normalize last tensor.
    @tensor new_A[i, s, j] := psi.A[1][i, s, k]*US[k, j]
    psi.A[1] = new_A./norm(new_A)

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
    psi.A[end] ./= sqrt(norm_psi)

    # Copy original state to make it the objective state to which we want to
    # approximate. Make it a new Mps.
    obj_A = deepcopy(psi.A)
    obj = Mps(obj_A, obj_A, psi.L, psi.d)
    # Reduce the bond dimension of the state to the wanted one.
    svd_truncate!(psi, D)

    # Create left and right environments.
    Le = fill(ones(T, 1, 1), psi.L)
    Re = fill(ones(T, 1, 1), psi.L)
    # Initialize left environment.
    for i=2:psi.L
        Le[i] = prop_right2(Le[i-1], psi.A[i-1], obj.A[i-1])
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
        @tensor Mi[l1, s, r1] := Le[i][l1, l2]*obj.A[i][l2, s, r2]*Re[i][r1, r2]
        Mi = conj(Mi)

        # Update left and right environments.
        if sense == +1
            Mi = reshape(Mi, (size(Le[i], 1)*psi.d, size(Re[i], 1)))
            Qa, Ra = qr(Mi)
            Qa = Matrix(Qa)
            Qa = reshape(Qa, (size(Le[i], 1), psi.d, size(Qa, 2)))

            # Update left and right environments at L[i+1] and R[i] and psi.
            psi.A[i] = Qa
            if i < psi.L
                Le[i+1] = prop_right2(Le[i], Qa, obj.A[i])
                Re[i] = Ra*Re[i]
            end
        else
            Mi = reshape(Mi, (size(Le[i], 1), psi.d*size(Re[i], 1)))
            La, Qa = lq(Mi)
            Qa = Matrix(Qa)
            Qa = reshape(Qa, (size(Qa, 1), psi.d, size(Re[i], 1)))

            # Update left and right environments at L[i] and R[i-1] and psi.
            psi.A[i] = Qa
            if i > 1
                Re[i-1] = prop_left2(Qa, obj.A[i], Re[i])
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
            h5write(filename, "A$i", psi.A[i])
        else
            h5write(filename, "realA$i", real(psi.A[i]))
            h5write(filename, "imagA$i", imag(psi.A[i]))
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
            psi.A[i] = h5read(filename, "A$i")
        else
            psi.A[i] = complex.(h5read(filename, "realA$i"),
                                h5read(filename, "imagA$i"))
        end
    end

    return psi
end
