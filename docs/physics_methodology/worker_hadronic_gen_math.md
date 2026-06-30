# Hadronic EoS Generation (src/physics/worker_hadronic_gen.py)

This file is responsible for creating the theoretical 'Hadronic' stars. These are stars whose cores are made of standard nuclear matter (neutrons, protons, hyperons).

I used spectral representations and piecewise polytropes to construct these EoS curves, drawing heavily from standard nuclear physics baselines like APR4 and SLy (Read et al., 2009). The script generates a baseline, slightly randomizes it to create variations (so the ML model has a lot of data to train on), and then passes it to the sequence solver.
