# Quark EoS Generation (src/physics/worker_quark_gen.py)

This file is responsible for generating 'Quark' stars. Unlike Hadronic stars, these stars have undergone a phase transition in their core, breaking down neutrons into a soup of up, down, and strange quarks.

To simulate this, I used the MIT Bag Model and Constant Speed of Sound (CSS) parameterizations (Alford et al., 2013). The script essentially takes a Hadronic crust, calculates a phase transition pressure, and then splices in the Quark matter core equations. This creates the distinct 'kink' in the mass-radius curves that our machine learning models try to detect!
