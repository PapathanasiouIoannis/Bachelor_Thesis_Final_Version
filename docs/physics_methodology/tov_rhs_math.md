# The Tolman-Oppenheimer-Volkoff Equations

The structural integration of the star relies on the Tolman-Oppenheimer-Volkoff (TOV) equations. I decoupled this entirely from the specific microphysics, so `tov_rhs.py` only handles the General Relativity part of the problem.

For a static spherically symmetric body, the mass gradient and pressure gradient are described by:

$$ \frac{dm}{dr} = 4\pi r^2 \epsilon(r) $$

$$ \frac{dP}{dr} = - \frac{G}{c^2} \frac{(\epsilon(r) + P(r))(m(r) + 4\pi r^3 P(r))}{r(r - \frac{2Gm(r)}{c^2})} $$

The variable $\epsilon(r)$ comes from whatever equation of state is currently being evaluated.

To get the tidal deformability $\Lambda$, the script also evaluates a Riccati equation for the metric perturbation $y(r)$, as derived by Hinderer (2008). The differential equation is:

$$ r \frac{dy}{dr} + y(r)^2 + y(r) F(r) + r^2 Q(r) = 0 $$

The terms $F(r)$ and $Q(r)$ are calculated using the local speed of sound $c_s^2$. Because $1/r$ causes numerical blowups at the exact center of the star, I implemented a Taylor expansion boundary condition when $r$ is extremely close to zero.
