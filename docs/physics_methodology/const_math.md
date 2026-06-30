# Constants and Configuration

The `src/config.py` file centralizes all the physical constants and hyperparameters for the project. 

The solvers require geometric units to keep the floating point numbers stable during Runge-Kutta integration. The fundamental constants used are:

- $M_\odot = 1.989 \times 10^{30} \, \text{kg}$
- $c = 2.9979 \times 10^8 \, \text{m/s}$
- $G = 6.6743 \times 10^{-11} \, \text{m}^3 \text{kg}^{-1} \text{s}^{-2}$
- $\hbar c = 197.33 \, \text{MeV fm}$

The mass conversion factor is roughly $1.4766$, and the pressure conversion is roughly $1.124 \times 10^{-5}$. These are strictly applied inside the TOV module.
