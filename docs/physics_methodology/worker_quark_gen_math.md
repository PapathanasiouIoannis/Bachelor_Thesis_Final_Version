# Generalized CFL Quark Generator (`src/physics/worker_quark_gen.py`)

For the exotic Quark matter class, the generation framework employs a **Generalized Color-Flavor Locked (CFL)** model rather than a simplistic MIT Bag parameterization (Alford et al., 2013). This properly models the effects of pairing gaps and strange quark mass within the quark core.

The model is parameterized by the pairing gap $\Delta$, the strange quark mass $m_s$, and the bag constant $B$. The energy density $\epsilon$ and pressure $P$ are evaluated as functions of the quark chemical potential $\mu$.

The key to the structural variation in these Quark stars is the effective gap squared:
$$ \Delta_{\text{eff}}^2 = \Delta^2 - \frac{m_s^2}{4} $$

The energy density expands as:
$$ \epsilon = \frac{3}{4\pi^2}\mu^4 + \frac{3}{\pi^2}\Delta_{\text{eff}}^2 \mu^2 + B $$

Crucially, the exact derivative $\frac{dP}{d\mu} / \frac{d\epsilon}{d\mu}$ yields the rigorous speed of sound squared $c_s^2$ for the CFL phase:
$$ c_s^2 = \frac{\mu^2 + 2\Delta_{\text{eff}}^2}{3\mu^2 + 2\Delta_{\text{eff}}^2} $$

This exact analytical formulation guarantees that at asymptotically high densities ($\mu \to \infty$), the speed of sound strictly approaches the conformal limit of $c_s^2 = \frac{1}{3}$, a key hallmark of asymptotic freedom in QCD.
