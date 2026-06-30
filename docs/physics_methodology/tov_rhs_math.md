# The Tolman-Oppenheimer-Volkoff (TOV) System (`src/physics/tov_rhs.py`)

The structure of a static, spherically symmetric, relativistic star is determined by integrating the Tolman-Oppenheimer-Volkoff (TOV) equations (Tolman, 1939; Oppenheimer & Volkoff, 1939). In this framework, the equations are decoupled from specific microphysics and strictly evaluate General Relativity.

The system of ordinary differential equations governing the enclosed mass $m(r)$ and the internal pressure $P(r)$ as a function of the radial coordinate $r$ is:

$$ \frac{dm}{dr} = 4\pi r^2 \epsilon(r) $$

$$ \frac{dP}{dr} = - \frac{G}{c^2} \frac{(\epsilon(r) + P(r))(m(r) + 4\pi r^3 P(r))}{r(r - \frac{2Gm(r)}{c^2})} $$

where $\epsilon(r)$ is the energy density supplied by the Equation of State (EoS).

Furthermore, the framework computes the dimensionless tidal deformability $\Lambda$. Following Hinderer (2008), this requires integrating an auxiliary Riccati equation for the metric perturbation $y(r)$:

$$ r \frac{dy}{dr} + y(r)^2 + y(r) F(r) + r^2 Q(r) = 0 $$

where $F(r)$ and $Q(r)$ are heavily dependent on the local speed of sound squared $c_s^2$. The script `tov_rhs.py` evaluates these derivatives and employs a regularized Taylor expansion at the stellar core ($r \approx 0$) to avert coordinate singularities.
