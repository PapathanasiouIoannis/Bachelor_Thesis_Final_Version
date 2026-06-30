# The Tolman-Oppenheimer-Volkoff System and Tidal Deformability

The core of the integration framework in `src/physics/tov_rhs.py` is the evaluation of the Tolman-Oppenheimer-Volkoff (TOV) equations. Originally derived by Tolman (1939) and Oppenheimer & Volkoff (1939), these equations describe the structure of a static, spherically symmetric body of isotropic material in General Relativity. 

I made a deliberate design choice to decouple this solver entirely from the specific microphysics (the Equation of State). The script only takes the local energy density $\epsilon$ and local pressure $P$ to calculate the structural gradients.

The differential equations governing the enclosed mass $m(r)$ and pressure $P(r)$ as you move outward radially are:

$$ \frac{dm}{dr} = 4\pi r^2 \epsilon(r) $$

$$ \frac{dP}{dr} = - \frac{G}{c^2} \frac{(\epsilon(r) + P(r))(m(r) + 4\pi r^3 P(r))}{r(r - \frac{2Gm(r)}{c^2})} $$

It's important to note the denominator in the pressure gradient. As $r$ approaches zero at the exact center of the star, the $1/r$ term introduces a coordinate singularity. If we tried to let the ODE solver handle this naturally, it would result in a `NaN` blowup. To fix this, I implemented a strict Taylor expansion boundary condition that takes over whenever $r < 10^{-4}$ km.

### Integrating the Metric Perturbation

Beyond just mass and radius, modern multi-messenger astronomy requires us to understand how the star deforms in a binary inspiral, which we measure via gravitational waves. Following the foundational work of Tanja Hinderer (2008), the tidal deformability $\Lambda$ can be derived by integrating an auxiliary Riccati equation alongside the TOV system.

This equation tracks $y(r)$, which is related to the $l=2$ metric perturbation:

$$ r \frac{dy}{dr} + y(r)^2 + y(r) F(r) + r^2 Q(r) = 0 $$

The coefficients $F(r)$ and $Q(r)$ heavily rely on the microscopic properties of the matter, specifically the local speed of sound squared $c_s^2 = \frac{dP}{d\epsilon}$. They are defined as:

$$ F(r) = \frac{1 - 4\pi G r^2 (\epsilon(r) - P(r))}{1 - \frac{2Gm(r)}{r}} $$

$$ Q(r) = \frac{4\pi G \left( 5\epsilon(r) + 9P(r) + \frac{\epsilon(r) + P(r)}{c_s^2} \right)}{1 - \frac{2Gm(r)}{r}} - \frac{6}{r^2 \left(1 - \frac{2Gm(r)}{r}\right)} $$

Because $c_s^2$ appears directly in the denominator of $Q(r)$, any numerical noise that pushes the speed of sound close to zero will violently destabilize the Riccati integration. To prevent this, `tov_rhs.py` clamps $c_s^2$ to a strict thermodynamic floor ($10^{-10}$).
