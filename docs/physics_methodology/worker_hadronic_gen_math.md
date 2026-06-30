# Anchored Speed-of-Sound Hadronic Model

Generating realistic hadronic matter is conceptually difficult because our understanding of Quantum Chromodynamics (QCD) breaks down at the extreme densities found in neutron star cores. Many older computational models rely on joining multiple piecewise polytropic segments together, but this approach severely limits the available phase space and creates unnatural, sharp transitions in the speed of sound.

Instead of polytropes, `src/physics/worker_hadronic_gen.py` implements an advanced "Anchored Speed-of-Sound" model. This technique generates a massive, unbiased theoretical dataset by mathematically perturbing trusted nuclear physics baselines.

### The Crust and Baseline Anchors
For the low-density regime, the code strictly uses the Douchin & Haensel (2001) SLy crust parameterization, which is widely considered the gold standard for the outer layers. As we move deeper, the code randomly anchors the core to one of over twenty peer-reviewed nuclear equations of state (such as APR4 or various MDI fits).

### The Gaussian Perturbation
At densities where the chosen baseline EoS starts to become theoretically uncertain, the code randomly splices in a smooth perturbation directly into the speed of sound squared ($c_s^2$). This is done by superimposing a Gaussian bump onto the baseline trajectory:

$$ c_s^2(\epsilon) = c_{s, \text{base}}^2(\epsilon) + A \exp\left(-\frac{1}{2}\left(\frac{\epsilon - \epsilon_0}{\sigma}\right)^2\right) $$

Here:
- $A$ is the randomized amplitude of the perturbation.
- $\epsilon_0$ is the target energy density where the perturbation is centered.
- $\sigma$ dictates the width of the Gaussian spread.

After constructing this highly varied $c_s^2$ profile, the framework enforces the strict physical constraints ($0 < c_s^2 \leq 1$). Because the TOV equations require the pressure $P$ rather than just the speed of sound, the script mathematically reconstructs the new pressure profile by explicitly integrating the relation $c_s^2 = \frac{dP}{d\epsilon}$:

$$ P(\epsilon) = \int_{\epsilon_{\text{trans}}}^{\epsilon} c_s^2(\epsilon') \, d\epsilon' + P_{\text{trans}} $$

This approach produces an incredibly diverse dataset of theoretically viable hadronic EoS curves, exhibiting complex, non-linear stiffening and softening behaviors that rigid polytropes simply cannot mimic.
