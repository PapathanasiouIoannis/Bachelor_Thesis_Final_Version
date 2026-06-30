# The Baseline EoS Library (src/physics/get_eos_library.py)

This is basically a dictionary that stores the standard, peer-reviewed Hadronic EoS models (like SLy, APR, BSk). 

I implemented this so the worker_hadronic_gen.py script has a solid, scientifically accurate foundation to build from. It loads the base thermodynamic arrays for these models, which are then perturbed to create our massive dataset.
