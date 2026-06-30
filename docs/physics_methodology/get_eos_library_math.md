# EoS Baseline Library

The `get_eos_library.py` script holds the algebraic parameterizations for the nuclear baselines used in the hadronic generator. These fits are mostly sourced from Read et al. (2009).

The library includes over twenty hadronic models and a specific multi-layer crust model from Douchin & Haensel. Because evaluating symbolic math in Python is slow, I use SymPy to define the equations but then `lambdify` them into fast NumPy C-functions. I added a caching mechanism so the recompilation only happens once when the parallel workers start up, which fixed a massive bottleneck in the data generation.
