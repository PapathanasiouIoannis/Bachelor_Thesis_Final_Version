# Solving the Core Sequence (src/physics/solve_sequence.py)

To actually simulate a whole family of neutron stars (a 'sequence'), we can't just run the TOV equations once. We have to sweep through different central pressures.

In solve_sequence.py, I wrote a solver that starts with a given EoS and steps through a range of central pressures. For each central pressure, it calls the TOV solver to integrate outwards. 

This process creates a complete mass-radius curve for a specific EoS model. I made sure to add strict physical filters here—if a star is causality-violating (the speed of sound is faster than light, ^2 > 1$) or if it's gravitationally unstable (/dho_c < 0$), the script completely throws it out.
