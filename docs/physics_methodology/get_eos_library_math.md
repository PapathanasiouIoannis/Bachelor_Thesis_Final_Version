# EoS Baseline Library and SymPy Optimization

The `src/physics/get_eos_library.py` script serves as the repository for all the analytical nuclear physics models used to anchor the hadronic core generator.

The baseline parameterizations are heavily sourced from the seminal work of Read et al. (2009), who successfully fit dozens of realistic nuclear EoS models (such as SLy, APR4, and various MDI configurations) into continuous analytical expressions. The script also includes the full multi-layer Douchin & Haensel parameterization for the outer crust.

### The Lambdify Bottleneck Fix
Initially, these algebraic fits were evaluated purely using SymPy to guarantee symbolic mathematical accuracy. However, this caused an immense computational bottleneck. When generating 100,000 equations of state across dozens of parallel CPU workers, constantly recompiling symbolic math trees slowed the pipeline to a crawl.

To fix this, I refactored the library to use a lazy-loaded global caching system. The script now uses SymPy's `lambdify` function to compile the symbolic expressions down into highly optimized, C-level NumPy functions. This compilation happens exactly once per process. Subsequent calls instantly retrieve the cached NumPy functions, speeding up the data generation phase by over two orders of magnitude while preserving exact mathematical precision.
