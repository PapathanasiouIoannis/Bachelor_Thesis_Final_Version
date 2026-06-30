# Physics Constants and Configuration (`src/config.py`)

All fundamental physical constraints and limits are strictly governed by the centralized configuration dictionary in `src/config.py`.

The framework operates in a geometrized unit system where necessary, converting standard nuclear units to maintain numerical stability during Runge-Kutta integration.

### Fundamental Constants
- $M_\odot = 1.989 \times 10^{30} \, \text{kg}$
- $c = 2.9979 \times 10^8 \, \text{m/s}$
- $G = 6.6743 \times 10^{-11} \, \text{m}^3 \text{kg}^{-1} \text{s}^{-2}$
- $\hbar c = 197.33 \, \text{MeV fm}$

The exact conversions for geometric mass ($1.4766$) and pressure ($1.124 \times 10^{-5}$) are strictly enforced across all solvers.
