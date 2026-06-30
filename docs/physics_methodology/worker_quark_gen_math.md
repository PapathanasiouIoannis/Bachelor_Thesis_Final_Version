# Generalized CFL Quark Model

For the quark star generation, the standard MIT Bag model is too simplistic to cover the phase space. `worker_quark_gen.py` implements the Generalized Color-Flavor Locked (CFL) phase equations (Alford et al. 2013). This accounts for quark pairing gaps and the strange quark mass.

The model is defined by the gap $\Delta$, strange mass $m_s$, and the bag constant $B$. The structural variation mostly depends on the effective gap squared, which I calculate as:

$$ \Delta_{\text{eff}}^2 = \Delta^2 - \frac{m_s^2}{4} $$

The energy density is evaluated from the chemical potential $\mu$:

$$ \epsilon = \frac{3}{4\pi^2}\mu^4 + \frac{3}{\pi^2}\Delta_{\text{eff}}^2 \mu^2 + B $$

By taking the derivative ratio $\frac{dP}{d\mu} / \frac{d\epsilon}{d\mu}$, the code calculates the exact speed of sound squared for the CFL phase:

$$ c_s^2 = \frac{\mu^2 + 2\Delta_{\text{eff}}^2}{3\mu^2 + 2\Delta_{\text{eff}}^2} $$

This formula is important because it mathematically guarantees that at extremely high densities, the speed of sound approaches the conformal limit of $1/3$, which is what we expect from asymptotic freedom in QCD.
