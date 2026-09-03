# Get started

String Amplitudes is a python package that aims to numerically compute string amplitudes. Currently, it supports the generation of ribbon graphs associated with the moduli space of Riemann surfaces. In a specific conformal frame associated with the ribbon graph structure, it supports the computation of holomorphic data, the free field partition functions, and the bc ghost correlation function using the formula of Verlinde and Verlinde. It also supports evaluation of the critical bosonic string integrand.

## Installation

Install as

```console
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

## Generation of ribbon graphs

Generate the genus two ribbon graphs with one face and inspect the boundary of the first graph:

```python
from string_amplitudes import generate_ribbon_graphs, get_boundary_data

graphs = generate_ribbon_graphs(genus=2)
ribbon_graph = graphs[0]
boundary_data = get_boundary_data(ribbon_graph)
```

The resulting boundary data and a set of positive edge lengths can then be used to compute a period matrix.

## Next steps

* [The full set of calculations presented in our paper](calculations)
* [the API reference for all classes and functions in the repository](api/index)
