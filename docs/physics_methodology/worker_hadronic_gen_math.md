# Anchored Speed-of-Sound Hadronic Generator (`src/physics/worker_hadronic_gen.py`)

To comprehensively sample the phase space of theoretical Hadronic matter, the framework abandons restrictive piecewise polytropes in the deep core. Instead, it utilizes an advanced **Anchored Speed-of-Sound** generator.

The generation process retains a scientifically rigorous analytic crust (Douchin & Haensel, 2001) and anchors the low-density core to a randomly selected nuclear baseline from a library of peer-reviewed EoS models (e.g., SLy, APR4). 

At extreme densities where the nuclear baseline diverges or loses reliability, the generator splices in a smooth, randomly perturbed speed-of-sound trajectory. This is achieved mathematically by superimposing a Gaussian bump onto the baseline speed of sound $c_{s,\text{base}}^2$:

$$ c_s^2(\epsilon) = c_{s, \text{base}}^2(\epsilon) + A \exp\left(-\frac{1}{2}\left(\frac{\epsilon - \epsilon_0}{\sigma}\right)^2\right) $$

where:
- $A$ is the perturbation amplitude.
- $\epsilon_0$ is the location of the bump in energy density space.
- $\sigma$ is the width of the perturbation.

The resulting continuous trajectory for $c_s^2$ is rigorously constrained by causality ($c_s^2 \leq 1$) and thermodynamic stability ($c_s^2 > 0$). The new pressure is then mathematically reconstructed by re-integrating $P = \int c_s^2 \, d\epsilon$. This produces a highly diverse, yet physically grounded, dataset of purely hadronic EoS curves.
