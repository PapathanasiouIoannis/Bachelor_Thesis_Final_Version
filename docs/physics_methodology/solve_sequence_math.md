# Sequence Integration and Macroscopic Limits (`src/physics/solve_sequence.py`)

The sequence solver iteratively integrates the TOV equations across a defined manifold of central pressures $P_c$ to generate macroscopic mass-radius-deformability ($M-R-\Lambda$) sequences. 

### Integration Strategy
The framework employs the Runge-Kutta method of order 5(4) (`RK45`) via SciPy's `solve_ivp`. Integration proceeds outward from the core until the zero-pressure boundary surface is intercepted, isolated by a terminal event function where $P(r) = 0$.

### Physical Stability Cutoffs
Not all evaluated central pressures produce physically viable macroscopic stars. The script strictly filters configurations that violate theoretical stability:
1. **Thermodynamic Stability:** The sequence rejects any EoS mapping where $\frac{dP}{d\epsilon} \leq 0$.
2. **Causality:** The microscopic speed of sound must not exceed the speed of light, ensuring $c_s^2 \leq 1$.
3. **Buchdahl Limit:** The compactness parameter $C = \frac{GM}{Rc^2}$ is bounded to $C < \frac{4}{9}$, beyond which the star would undergo inevitable gravitational collapse into a black hole.

### The Tidal Love Number
Upon locating the surface boundary at $R$, the integration of the Riccati variable yields $y_R = y(R)$. This value is passed into the algebraic expression for the $l=2$ tidal Love number $k_2$:

$$ k_2 = \frac{8 C^5}{5}(1-2C)^2 [2C(y_R - 1) - y_R + 2] \Big\{ 2C [6 - 3y_R + 3C(5y_R - 8)] + 4C^3 [13 - 11y_R + C(3y_R - 2) + 2C^2(1+y_R)] + 3(1-2C)^2 [2 - y_R + 2C(y_R - 1)] \ln(1-2C) \Big\}^{-1} $$

The dimensionless tidal deformability is ultimately extracted as $\Lambda = \frac{2}{3} k_2 C^{-5}$.
