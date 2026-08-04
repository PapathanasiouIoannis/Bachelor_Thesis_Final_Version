# Repository-local EoS surrogate library

`src/physics/get_eos_library.py` compiles a local collection of analytic
energy-density functions \(\epsilon(P)\) and their derivatives. SymPy derives
\(d\epsilon/dP\), and `lambdify` caches NumPy callables once per process.

## Provenance boundary

The exact two-power and exponential coefficients in this file have not been
verified against a primary publication. They are not the piecewise-polytrope
coefficients published by [Read et al. (2009)](https://arxiv.org/abs/0812.2163).
The local four-part crust formula also has not been established as the published
analytic SLy representation of
[Haensel and Potekhin (2004)](https://arxiv.org/abs/astro-ph/0408324).

The controlled experiment therefore calls the selected model the **repository
APR-1 surrogate**. It must not be identified with Read `APR1` or the CompOSE
`APR` table. The naming distinctions and primary sources are documented in
[`docs/CONTROLLED_EOS_SWEEP.md`](../CONTROLLED_EOS_SWEEP.md).

## Controlled use

Production generation selects only the `APR-1` entry. The framework applies the
declared density shift at \(P_t=0.184\;\mathrm{MeV\,fm^{-3}}\), evaluates the
baseline sound speed

\[
c_s^2(P)=\left(\frac{d\epsilon}{dP}\right)^{-1},
\]

and passes it to the shared Gaussian deformation and pressure-reconstruction
path in `framework/eos_sweep.py`. The remaining entries stay available for
legacy diagnostics; they are not sampled by the controlled generator.
