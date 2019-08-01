# MPStates.jl
Matrix Product States algorithms in Julia.

## Description
This library intends to be a simple but performant implementation of several Matrix Product States (MPS) algorithms in pure Julia.  The implemented algortihms so far are the well known [1-site and 2-site DMRG](https://arxiv.org/abs/1008.3477) algorithms and [Strictly Single-Site DMRG](https://arxiv.org/abs/1501.05504). Functions to build general Matrix Product Operators (MPO) and to extract expectation values from MPS are also available.

## Installation
Once you have Julia installed you will need the following packages: `TensorOperations` (for tensor contractions using Einsum notation), `HDF5` to store an `Mps`, and `Arpack` to locally minimize the energy of a state. You can do this in Julia with:
```julia
using Pkg
Pkg.add("HDF5")
Pkg.add("TensorOperations")
Pkg.add("Arpack") # This is difficult to build if Julia was built from source using MKL.
```

After the packages are installed you have to download this repository (remove the .jl from the directory's name) and write at the top of your Julia script: 
```julia
push!(LOAD_PATH, "/full/path/to/MPStates/directory")
using MPStates
```

## Usage
See here [examples](https://github.com/alvarorga/MPStates.jl/tree/master/examples) of how to use the `Mps` and `Mpo` structures and how to find the ground state energy of a Hamiltonian.

WARNING: the API is yet unstable (but slowly converging).
