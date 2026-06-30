# Baseline EoS Library (`src/physics/get_eos_library.py`)

The baseline nuclear models acting as anchors for the hadronic generator are sourced from the extensive piecewise compilations of Read et al. (2009).

This library provides exact analytic algebraic fits for over twenty peer-reviewed hadronic models (including SLy, APR4, and the MDI family), alongside a highly accurate multi-layer crust model derived from Douchin & Haensel (2001). The script compiles these symbolic mathematical representations into highly optimized C-functions using NumPy and SymPy to prevent significant overhead bottlenecks during parallel data generation.
