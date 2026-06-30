# Sequence Integration

To get a full mass-radius curve, `solve_sequence.py` sweeps across a grid of central pressures and solves the TOV equations for each one. The integration uses SciPy's `solve_ivp` with the RK45 method. It steps outward radially and stops when the pressure drops to zero, which defines the star's surface.

Not every central pressure yields a physical star. I implemented a few strict limits that discard unphysical results:
- The speed of sound can not exceed light speed ($c_s^2 \le 1$).
- The pressure must increase with density ($\frac{dP}{d\epsilon} > 0$), or else it violates thermodynamic stability.
- The star's compactness $C = \frac{GM}{Rc^2}$ is bounded by the Buchdahl limit $C < \frac{4}{9}$. Anything denser would collapse into a black hole before we observe it.

When the surface is reached, we get the final value $y_R$. This is used to calculate the $l=2$ tidal Love number $k_2$:

$$ k_2 = \frac{8 C^5}{5}(1-2C)^2 [2C(y_R - 1) - y_R + 2] \Big\{ 2C [6 - 3y_R + 3C(5y_R - 8)] + 4C^3 [13 - 11y_R + C(3y_R - 2) + 2C^2(1+y_R)] + 3(1-2C)^2 [2 - y_R + 2C(y_R - 1)] \ln(1-2C) \Big\}^{-1} $$

Finally the dimensionless deformability is just $\Lambda = \frac{2}{3} k_2 C^{-5}$.
