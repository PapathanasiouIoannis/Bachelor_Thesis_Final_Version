# Anchored Speed-of-Sound Hadronic Model

Generating realistic hadronic matter is difficult because we don't know the exact physics at extreme core densities. Instead of using generic piecewise polytropes, `worker_hadronic_gen.py` uses an anchored speed-of-sound methodology.

The code uses a known analytical crust model and attaches it to a randomly selected nuclear baseline EoS (like APR4 or SLy) for the outer core. At higher densities where the baseline becomes uncertain, a random Gaussian perturbation is injected directly into the speed of sound squared:

$$ c_s^2(\epsilon) = c_{s, \text{base}}^2(\epsilon) + A \exp\left(-\frac{1}{2}\left(\frac{\epsilon - \epsilon_0}{\sigma}\right)^2\right) $$

Here $A$ is a random amplitude, $\epsilon_0$ is the center of the perturbation, and $\sigma$ controls the width. 

After generating this randomized $c_s^2$ path, the code checks that it stays causal ($c_s^2 \le 1$) and positive. The corresponding pressure is found by simply integrating $P = \int c_s^2 \, d\epsilon$. This creates a highly varied but physically valid dataset of hadronic equations of state without being constrained to rigid polytropic segments.
