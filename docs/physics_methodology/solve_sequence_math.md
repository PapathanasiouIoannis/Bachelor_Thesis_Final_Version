# Sequence Integration and Macroscopic Limits

To generate a full stellar sequence (the mass-radius curve), `src/physics/solve_sequence.py` sweeps over a defined grid of central pressures $P_c$ and integrates the TOV system outward for each one. 

The actual numerical integration relies on SciPy's `solve_ivp` utilizing the Runge-Kutta method of order 5(4) (`RK45`). The integration starts near $r=0$ and steps radially outward. It terminates dynamically using an event function triggered when the pressure drops to the surface cutoff threshold ($10^{-13}$ MeV/fm$^3$).

### Rigorous Physical Cutoffs

Not all central pressures yield stars that could physically exist in the universe. I implemented several hard theoretical filters to reject mathematically valid but physically impossible integration branches:

1. **Causality:** The speed of sound inside the star cannot exceed the speed of light in a vacuum. If the microscopic generator returns an equation of state where $c_s^2 > 1$, the sequence is immediately discarded.
2. **Thermodynamic Stability:** The pressure must strictly monotonically increase with energy density, meaning $\frac{dP}{d\epsilon} > 0$. An EoS that violates this would be mechanically unstable to collapse.
3. **The Buchdahl Limit:** General relativity dictates a maximum theoretical compactness $C = \frac{GM}{Rc^2}$. If a star's compactness exceeds the Buchdahl limit of $C = \frac{4}{9}$, it would inevitably undergo total gravitational collapse into a black hole.

### Extracting the Tidal Love Number

When the `RK45` solver successfully reaches the surface radius $R$, it extracts the final value of the Riccati variable, $y_R = y(R)$. This value is the key to calculating how the star responds to external tidal fields. 

Following Hinderer (2008), $y_R$ is substituted into a massive algebraic expression for the $l=2$ tidal Love number $k_2$. The equation is highly sensitive to the compactness $C$:

$$ k_2 = \frac{8 C^5}{5}(1-2C)^2 [2C(y_R - 1) - y_R + 2] \Big\{ 2C [6 - 3y_R + 3C(5y_R - 8)] + 4C^3 [13 - 11y_R + C(3y_R - 2) + 2C^2(1+y_R)] + 3(1-2C)^2 [2 - y_R + 2C(y_R - 1)] \ln(1-2C) \Big\}^{-1} $$

Once $k_2$ is isolated, it is trivial to calculate the dimensionless tidal deformability $\Lambda$, which is the actual parameter constrained by gravitational wave observatories like LIGO during events like GW170817:

$$ \Lambda = \frac{2}{3} k_2 C^{-5} $$
