# Controlled APR-1-Surrogate / CFL4 EoS Sweep

## Purpose and claim boundary

This experiment generates a deterministic, paired sweep over one repository
hadronic baseline and one fixed quark-matter baseline. It is designed to test
**APR-1-surrogate versus fixed-CFL4 model-pair discrimination** under one shared
Gaussian sound-speed deformation family.

The experiment does **not** provide enough model diversity to establish general
hadronic-versus-quark phase classification. Because there is only one baseline
per label, matter class and baseline identity remain perfectly confounded. A
reported classifier result must therefore retain the model-pair wording above.

## Fixed experiment definition

| Quantity | Controlled value | Status |
|---|---:|---|
| Hadronic baseline | repository `APR-1` analytic surrogate only | fixed |
| Quark baseline | analytic CFL MIT-bag model, CFL4 tuple | fixed |
| Bag energy density, $B$ | $60\ \mathrm{MeV\,fm^{-3}}$ | literature benchmark |
| Pairing gap, $\Delta$ | $100\ \mathrm{MeV}$ | literature benchmark |
| Strange-quark mass, $m_s$ | $150\ \mathrm{MeV}$ | literature benchmark |
| Gaussian location, $\epsilon_0$ | $220\ \mathrm{MeV\,fm^{-3}}$ | project choice |
| Gaussian width, $\sigma$ | $50\ \mathrm{MeV\,fm^{-3}}$ | project choice |
| Gaussian amplitude, $A$ | $-0.05,-0.04,\ldots,0.08,0.09$ | project sweep |

There are 15 amplitudes with a step of $0.01$, including the undeformed
$A=0$ control. For every amplitude, the generators attempt exactly one
hadronic curve and one quark curve with the same `Sweep_ID`. The controlled
values live in `src/config.py`; the common construction is implemented in
`framework/eos_sweep.py` and called by both production workers.

The CFL4 tuple is the parameter set named CFL4 by Vásquez Flores and Lugones.
Here $B$ is a bag **energy density** in $\mathrm{MeV\,fm^{-3}}$, not
$B^{1/4}$ in MeV. The values of $\epsilon_0$, $\sigma$, and the sampled $A$
values are experimental controls chosen by this project; they are not APR or
CFL parameters inferred from the cited literature.

## Hadronic baseline identity and provenance warning

The repository baseline called `APR-1` is the analytic expression

$$
\epsilon(P)
=0.000719964\,P^{1.85898}
+108.975\,P^{0.340074},
$$

with $P$ and $\epsilon$ in $\mathrm{MeV\,fm^{-3}}$. The controlled builder
uses only this baseline. At the configured crust/core transition pressure it
subtracts a constant energy-density offset from the analytic core so that the
core and crust have the same $\epsilon$ at the anchor. This repository-specific
surrogate and matching convention must be recorded with generated artifacts.

The name `APR-1` is not unambiguous across the literature:

- Koliogiannis and Moustakidis use **APR-1** for the Akmal
  A18+$\delta v$-free UIX interaction, A18+UIX, and **APR-2** for
  A18+$\delta v$+UIX*.
- Read et al. use `APR1` for a particular piecewise-polytropic fit. Their fit is
  not the repository's two-power expression.
- The CompOSE table named `APR` uses A18+$\delta v$+UIX*, which corresponds to
  the interaction called APR-2 in the Koliogiannis--Moustakidis convention,
  not their APR-1.

The exact two-power coefficients above are not given in the primary Akmal or
Read papers. A verbatim match was located only in a later secondary thesis,
without an independently documented primary fit procedure, validity interval,
or fit uncertainty. Consequently, this documentation calls it the
**repository APR-1 surrogate** and does not claim that it is a verified Read,
CompOSE, or tabulated Akmal EoS.

## Shared Gaussian deformation

For either baseline, the framework constructs

$$
g(\epsilon)=
\exp\!\left[-\frac{1}{2}
\left(\frac{\epsilon-\epsilon_0}{\sigma}\right)^2\right],
\qquad
c_{s,A}^2(\epsilon)=c_{s,0}^2(\epsilon)+A g(\epsilon).
$$

$A$ is dimensionless. The same numerical $(A,\epsilon_0,\sigma)$ is used for
both members of each pair. This does not mean that the deformation probes an
identical microscopic regime: the hadronic coordinate is evaluated after the
repository's crust/core energy-density matching, whereas the bare CFL4 EoS
begins at its finite self-bound surface density.

### Causality and the selected amplitude support

The framework derives each baseline's admissible interval from

$$
0<c_{s,A}^2(\epsilon)\leq1.
$$

Since $g(\epsilon)>0$ on its numerical support, this is equivalent to

$$
A>
\max_\epsilon\!\left[-\frac{c_{s,0}^2(\epsilon)}{g(\epsilon)}\right],
\qquad
A\leq
\min_\epsilon\!\left[\frac{1-c_{s,0}^2(\epsilon)}{g(\epsilon)}\right].
$$

For the current grids and conventions, the approximate intervals are

$$
\begin{aligned}
\text{repository APR-1 surrogate:}&\quad
-0.10875 < A \leq 0.87916,\\
\text{fixed CFL4:}&\quad
-0.35977 < A \leq 0.64020.
\end{aligned}
$$

Their causal and thermodynamically stable intersection is therefore
approximately

$$
-0.10875 < A \leq 0.64020.
$$

This intersection is necessary but not sufficient for an accepted stellar
sequence. Every member is also run through the repository's TOV solver and
must satisfy the current $M_{\max}\geq2.08\,M_\odot$ validation. Trial sweeps
show that the softened CFL4 sequence reaches this rejection boundary near
$A\lesssim-0.075$. The default lower endpoint $A=-0.05$ deliberately keeps a
margin from that solver- and tolerance-dependent boundary. The upper endpoint
$A=0.09$ retains the intended small-deformation study scale; it is not the
fundamental causal upper limit.

Thus `[-0.05, 0.09]` is the default **validated experimental grid**, not a
universal literature interval. Any change to a baseline, grid, stellar
acceptance threshold, or numerical tolerance requires the interval and all TOV
sequences to be recomputed.

## Thermodynamic reconstruction and failure behavior

Changing $c_s^2$ while retaining the original $\epsilon(P)$ would violate
$c_s^2=dP/d\epsilon$. The framework therefore reconstructs pressure after the
deformation:

$$
P_A(\epsilon)=P_{\mathrm{anchor}}
+\int_{\epsilon_{\mathrm{anchor}}}^{\epsilon}
c_{s,A}^2(u)\,du.
$$

For the hadronic EoS, the anchor is the fixed crust/core transition. For CFL4,
it is the self-bound surface at $P=0$. The implementation uses cumulative
trapezoidal integration and monotone PCHIP interpolation to recover
$\epsilon_A(P)$ and $c_{s,A}^2(P)$.

The controlled path performs **no clipping** of negative, zero, or superluminal
sound speeds. It validates the requested amplitude before generation and
retains only the stable causal prefix when the baseline reaches its first
high-density causal endpoint. An invalid deformation, an insufficient causal
prefix, or a failed stellar acceptance check causes generation to fail rather
than silently replacing $c_s^2$ with a floor or cap.

## Primary and authoritative references

1. A. Akmal, V. R. Pandharipande, and D. G. Ravenhall, “Equation of state of
   nucleon matter and neutron star structure,” *Physical Review C* **58**,
   1804 (1998), [arXiv:nucl-th/9804027](https://arxiv.org/abs/nucl-th/9804027),
   [doi:10.1103/PhysRevC.58.1804](https://doi.org/10.1103/PhysRevC.58.1804).
2. P. S. Koliogiannis and Ch. C. Moustakidis, “Effects of the equation of
   state on the bulk properties of maximally rotating neutron stars,”
   *Physical Review C* **101**, 015805 (2020),
   [arXiv:1907.13375](https://arxiv.org/abs/1907.13375),
   [doi:10.1103/PhysRevC.101.015805](https://doi.org/10.1103/PhysRevC.101.015805).
3. J. S. Read et al., “Constraints on a phenomenologically parametrized
   neutron-star equation of state,” *Physical Review D* **79**, 124032 (2009),
   [arXiv:0812.2163](https://arxiv.org/abs/0812.2163),
   [doi:10.1103/PhysRevD.79.124032](https://doi.org/10.1103/PhysRevD.79.124032).
4. G. Lugones and J. E. Horvath, “Color-flavor locked strange matter,”
   *Physical Review D* **66**, 074017 (2002),
   [arXiv:hep-ph/0211070](https://arxiv.org/abs/hep-ph/0211070),
   [doi:10.1103/PhysRevD.66.074017](https://doi.org/10.1103/PhysRevD.66.074017).
5. C. Vásquez Flores and G. Lugones, “Constraining color flavor locked strange
   stars in the gravitational wave era,” *Physical Review C* **95**, 025808
   (2017), [arXiv:1702.02081](https://arxiv.org/abs/1702.02081),
   [doi:10.1103/PhysRevC.95.025808](https://doi.org/10.1103/PhysRevC.95.025808).
6. E. Fonseca et al., “Refined Mass and Geometric Measurements of the High-mass
   PSR J0740+6620,” *The Astrophysical Journal Letters* **915**, L12 (2021),
   [arXiv:2104.00880](https://arxiv.org/abs/2104.00880),
   [doi:10.3847/2041-8213/ac03b8](https://doi.org/10.3847/2041-8213/ac03b8).

The [CompOSE APR entry](https://compose.obspm.fr/eos/68) is included as an
authoritative database cross-check for the APR naming distinction; it is not
the source of the repository surrogate's analytic coefficients. The secondary
verbatim coefficient match noted above is
[arXiv:2509.13037](https://arxiv.org/abs/2509.13037), and is not treated as a
primary validation source.
