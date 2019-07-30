# MPStates.jl
Matrix Product States algorithms in Julia.

## Description
This library intends to be a simple but performant implementation of several Matrix Product States (MPS) algorithms in pure Julia.  The implemented algortihms so far are the well known [1-site and 2-site DMRG](https://arxiv.org/abs/1008.3477) algorithms and [Strictly Single-Site DMRG](https://arxiv.org/abs/1501.05504). Functions to build general Matrix Product Operators (MPO) and to extract expectation values from MPS are also available.

## Usage

(This will be moved to a Jupyter notebook when the API is more stable)

We start by creating an `Mps` which will be minimized to compute the ground state of some Hamiltonian
```julia
using MPStates, LinearAlgebra

# Number of sites of the Mps.
L = 10
# Type of the Mps.
T = Float64
# Physical dimension.
d = 2

psi = init_mps(Float64, L, "random", d)
```

We can already start to measure some observables of the `Mps` using the `expected` function (no need to remember more names thanks to Julia's multiple dispatch!)
```julia
# Compute the occupation number at site 5. We can either input the observable (operator) as
# a string or as a full matrix. 
ex1 = expected(psi, "n", 5)
ex2 = expected(psi, [[0. 0.]; [0. 1.]], 5)
# ex1 == ex2

# Correlations can be easily measured too!
ex3 = expected(psi, "b+", 1, "b", 3)
ex4 = expected(psi, [[0. 1.]; [0. 0.]], 1, [[0. 0.]; [1. 0.]], 3)
# ex3 == ex4

# If the operators are fermionic creation and annihilation operators we have to indicate it
# in the `ferm_op` argument, which is just the fermionic parity operator `I - 2n`.
ex5 = expected(psi, "c+", 1, "c", 3, ferm_op="Z")
ex6 = expected(psi, [[0. 1.]; [0. 0.]], 1, [[0. 0.]; [1. 0.]], 3, ferm_op=[[1. 0.]; [0. -1.]])
# ex5 == ex6

# When two operators act on the same site, `expected` multiplies them and makes them act on
# that site, so that
ex7 = expected(psi, "c+", 3, "c", 3)
# and 
ex8 = expected(psi, "n", 3)
# are equivalent.
```

Implemented string operators are for `d=2`: `"n"`, occupation operator; `"Z"`, fermionic parity operator, `I(2) - 2n`; `"a+"`, `"b+"`, `"c+"`, particle creation operators, they all share the same matrix representation, the distinction between fermions or hard-core bosons is made through the `ferm_op` argument; `"a"`, `"b"`, `"c"`, particle annihilation operators. For `d=3` (warning: not enough tested yet): `"Sz"`, `"S+"`, and `"S-"`.

Once we have defined the MPS, we want to build the Hamiltonian. This can be done in a very general way using the `add_ops!` function, with a notation similar to `expected`
```julia
# Make a Hubbard Hamiltonian (with open boundary conditions), let's first define the matrices 
# that describe the Hamiltonian.
# Hopping matrix.
J = 0.5*Symmetric(diagm(1 => ones(L-1)))
# Interaction matrix.
V = 0.7*diagm(1 => ones(L-1))

# Build the Hamiltonian.
H = init_mpo(T, L, d)
add_ops!(H, "c+", "c", J, ferm_op="Z")
add_ops!(H, "n", "n", V)
# Similar to `expected`, we could have writen: add_ops!(H, [[0. 0.]; [0. 1.]], [[0. 0.]; [0. 1.]], V)
```
This way of building the MPO with `add_ops!` is still a bit inefficient in which it creates a big array of tensors with many zero elements, which will slow down the DMRG algorithms. This will be fixed soon. However, by now you can use `H = init_mpo(T, J, V, is_fermionic=is_fermionic)` to create an MPO that is more efficient in the storage of the elements, but it is only valid for `d=2` and for interactions and hoppings as described above. This function will be deprecated when a more efficient version of `add_ops` is available.

Let's go to finding ground states, we do that with the `minimize!` function:
```julia
# Maximum bond dimension of the minimized Mps.
m = 64
# Algorithm: "DMRG1", "DMRG2", or "DMRG3S".
algorithm = "DMRG3S"
# Debug. Prints useful information about the convergence of the algorithm. Integer value.
debug = 1

# Find the ground state, which will be stored in `psi`. `E` and `var` are arrays that store
# the energy and variance of the state after every sweep of the algorithm.
E, var = minimize!(psi, H, m, algorithm, debug=debug)
```

WARNING: the API is yet unstable (but slowly converging).
