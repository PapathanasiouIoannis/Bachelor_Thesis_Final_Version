# Analytic CFL MIT-Bag Quark EoS

The controlled quark-star generator uses the zero-temperature, analytic
color-flavor-locked (CFL) extension of the MIT bag model, truncated at
order $O(m_s^2,\Delta^2)$. It is therefore more precise to call this the
**analytic CFL MIT-bag model** than a "generalized CFL" model. The
implementation describes bare, self-bound CFL strange stars; it does not
construct a hadron--quark hybrid EoS or attach a hadronic crust.

The equations below follow Lugones and Horvath's analytic approximation
[[1]](#references). The deterministic Gaussian sweep is constructed by the
shared methods in `framework/eos_sweep.py` and then passed to the common TOV
and tidal solver.

## Fixed CFL4 benchmark

The controlled experiment fixes one published parameter tuple rather than
randomly sampling quark microphysics:

| Parameter | Fixed value | Meaning |
|---|---:|---|
| $B$ | $60\ \mathrm{MeV\,fm^{-3}}$ | Bag energy density |
| $\Delta$ | $100\ \mathrm{MeV}$ | Constant CFL pairing gap |
| $m_s$ | $150\ \mathrm{MeV}$ | Constant strange-quark mass |

This is the parameter set named **CFL4** by Vásquez Flores and Lugones
[[2]](#references). For their implementation of the same analytic EoS, they
report $M_{\max}=2.127\,M_\odot$ and a corresponding radius of $11.41$ km.
Those published stellar values are reference checks, not substitutes for
validation with this repository's own grid and tolerances.

Here $B$ means the energy density in $\mathrm{MeV\,fm^{-3}}$, not
$B^{1/4}$. With the repository value $\hbar c=197.33\ \mathrm{MeV\,fm}$,

$$
B^{1/4}[\mathrm{MeV}]
=\left(B[\mathrm{MeV\,fm^{-3}}](\hbar c)^3\right)^{1/4},
$$

so $B=60\ \mathrm{MeV\,fm^{-3}}$ corresponds to
$B^{1/4}\simeq146.53\ \mathrm{MeV}$.

## Baseline thermodynamics

The pairing and strange-mass contributions can be grouped as

$$
q \equiv \Delta_{\mathrm{eff}}^2
=\Delta^2-\frac{m_s^2}{4}.
$$

For quark chemical potential $\mu$, the pressure and energy density are

$$
P_0(\mu)
=\frac{3\mu^4}{4\pi^2}
+\frac{3q\mu^2}{\pi^2}-B,
$$

$$
\epsilon_0(\mu)
=\frac{9\mu^4}{4\pi^2}
+\frac{3q\mu^2}{\pi^2}+B.
$$

The factor multiplying the quartic term in $\epsilon_0$ is
$9/(4\pi^2)$, not $3/(4\pi^2)$. This follows both from
$\epsilon=3\mu n_B-P$ and from the published CFL expansion [[1]](#references).

Writing

$$
a=\frac{3}{4\pi^2},\qquad b=\frac{3q}{\pi^2},\qquad x=\mu^2,
$$

gives $P_0=ax^2+bx-B$. For a requested pressure, the physical root is

$$
\mu^2=\frac{-b+\sqrt{b^2+4a(P+B)}}{2a}.
$$

The sound speed is evaluated from the same barotrope, not as an independent
quantity:

$$
c_{s,0}^2
=\frac{dP_0/d\mu}{d\epsilon_0/d\mu}
=\frac{\mu^2+2q}{3\mu^2+2q}.
$$

Consequently, $c_{s,0}^2\rightarrow1/3$ as $\mu\rightarrow\infty$. This is
the conformal asymptote of this truncated model; it should not be interpreted
as proof that the phenomenological bag model is quantitatively accurate at all
stellar densities.

## Unit convention

The public inputs and generated tables use

- $P$, $\epsilon$, and $B$ in $\mathrm{MeV\,fm^{-3}}$;
- $\mu$, $\Delta$, and $m_s$ in MeV;
- $c_s^2$ and the deformation amplitude $A$ as dimensionless quantities.

Internally, the analytic root is evaluated in inverse-femtometre natural
units:

$$
P_{\rm fm^{-4}}=\frac{P_{\rm MeV/fm^3}}{\hbar c},\quad
B_{\rm fm^{-4}}=\frac{B_{\rm MeV/fm^3}}{\hbar c},\quad
\mu_{\rm fm^{-1}}=\frac{\mu_{\rm MeV}}{\hbar c},
$$

with identical conversions for $\Delta$ and $m_s$. Multiplying the resulting
energy density in $\mathrm{fm^{-4}}$ by $\hbar c$ returns
$\mathrm{MeV\,fm^{-3}}$.

## Self-bound surface and absolute stability

The stellar surface occurs at $P_0=0$ but at a finite energy density
$\epsilon_s$. At zero temperature the CFL energy per baryon at this surface is

$$
\left.\frac{E}{A}\right|_{P=0}=3\mu_s.
$$

The absolute-stability test used for the benchmark is $3\mu_s\leq939$ MeV
[[1,2]](#references). Evaluating the repository equations for CFL4 gives,
approximately,

$$
\mu_s=263.88\ \mathrm{MeV},\qquad
\frac{E}{A}=791.63\ \mathrm{MeV},\qquad
\epsilon_s=215.90\ \mathrm{MeV\,fm^{-3}}.
$$

Thus the fixed baseline lies within the model's absolute-stability window.
The nonzero $\epsilon_s$ must be retained in both the EoS callable and the
tidal boundary condition.

## Gaussian sound-speed sweep

The sweep fixes

$$
\epsilon_0=220\ \mathrm{MeV\,fm^{-3}},\qquad
\sigma=50\ \mathrm{MeV\,fm^{-3}},
$$

and varies only the dimensionless amplitude $A$:

$$
g(\epsilon)=
\exp\left[-\frac{1}{2}
\left(\frac{\epsilon-\epsilon_0}{\sigma}\right)^2\right],
$$

$$
c_{s,A}^2(\epsilon)=c_{s,0}^2(\epsilon)+A g(\epsilon).
$$

An amplitude is admissible only when
$0<c_{s,A}^2\leq1$ over the complete retained EoS grid. Equivalently,

$$
A>\max_\epsilon\left[-\frac{c_{s,0}^2(\epsilon)}{g(\epsilon)}\right],
\qquad
A\leq\min_\epsilon\left[\frac{1-c_{s,0}^2(\epsilon)}{g(\epsilon)}\right].
$$

For the fixed CFL4 baseline and the stated Gaussian parameters, direct
evaluation gives the approximate strict interval

$$
-0.35977 < A \leq 0.64020.
$$

The experiment must use the intersection of this interval and the independently
derived APR-1 interval. Invalid values should be rejected before generation;
clipping $c_s^2$ would change the intended deformation.

### Pressure reconstruction

Changing the sound speed without changing $\epsilon(P)$ would violate
$c_s^2=dP/d\epsilon$. The framework therefore reconstructs pressure from the
deformed derivative, anchored at the self-bound surface:

$$
P_A(\epsilon_s)=0,
$$

$$
P_A(\epsilon)
=\int_{\epsilon_s}^{\epsilon}c_{s,A}^2(u)\,du.
$$

For the Gaussian contribution this may also be written analytically as

$$
P_A(\epsilon)=P_0(\epsilon)
+A\sigma\sqrt{\frac{\pi}{2}}
\left[
\operatorname{erf}\left(\frac{\epsilon-\epsilon_0}{\sqrt{2}\sigma}\right)
-\operatorname{erf}\left(\frac{\epsilon_s-\epsilon_0}{\sqrt{2}\sigma}\right)
\right].
$$

The implementation performs the equivalent cumulative trapezoidal integration
on the common energy-density grid and builds monotone PCHIP interpolators for
$\epsilon_A(P)$ and $c_{s,A}^2(P)$.

For CFL4, $\epsilon_s\simeq215.90\ \mathrm{MeV\,fm^{-3}}$, so the requested
Gaussian is already about $99.7\%$ of its peak value at the surface. The sweep
therefore probes surface and low-density CFL behavior as well as the stellar
interior; it is not exclusively a deep-core perturbation.

## Tidal surface-density correction

A self-bound star has a finite density immediately inside the zero-pressure
surface and vacuum immediately outside it. The tidal variable must therefore
be corrected across that discontinuity before computing the Love number. In
the repository's mixed units, `solve_sequence.py` applies

$$
y_R^{\rm corrected}
=y_R-\frac{G_{\rm conv}R^3\epsilon_s}{M}.
$$

The framework attaches the reconstructed EoS's unchanged anchor density
$\epsilon_s$ to the callable, ensuring that this correction remains active for
every amplitude. This treatment follows the self-bound-star boundary analysis
of Postnikov, Prakash, and Lattimer [[3]](#references).

## Scope and literature caveats

- The model is phenomenological. It omits perturbative-QCD corrections,
  vector interactions, a density-dependent gap, and higher-order strange-mass
  terms.
- The expansion assumes that $m_s/\mu$ and $\Delta/\mu$ are controlled.
  $m_s=150$ MeV is near the upper range for which Lugones and Horvath describe
  the $O(m_s^2)$ approximation as accurate [[1]](#references).
- Increasing a fixed $\Delta$ stiffens this MIT-bag parametrization, but that
  behavior is not reproduced generically by self-consistent NJL calculations.
  Very large fixed gaps should not be used merely to force a desired maximum
  mass [[4]](#references).
- The published CFL4 maximum mass is only modestly above the repository's
  $2.08\,M_\odot$ acceptance threshold. Each softened sweep member must be
  validated independently rather than inheriting the baseline's status.
- A numerically identical $(A,\epsilon_0,\sigma)$ does not guarantee that the
  deformation targets an identical physical regime in APR-1 and CFL4. The
  baseline-specific density and surface locations must be reported with the
  generated data.

## References

1. G. Lugones and J. E. Horvath, "Color-flavor locked strange matter,"
   *Physical Review D* **66**, 074017 (2002),
   [arXiv:hep-ph/0211070](https://arxiv.org/abs/hep-ph/0211070),
   [doi:10.1103/PhysRevD.66.074017](https://doi.org/10.1103/PhysRevD.66.074017).
2. C. Vásquez Flores and G. Lugones, "Constraining color flavor locked strange
   stars in the gravitational wave era," *Physical Review C* **95**, 025808
   (2017), [arXiv:1702.02081](https://arxiv.org/abs/1702.02081),
   [doi:10.1103/PhysRevC.95.025808](https://doi.org/10.1103/PhysRevC.95.025808).
3. S. Postnikov, M. Prakash, and J. M. Lattimer, "Tidal Love Numbers of Neutron
   and Self-Bound Quark Stars," *Physical Review D* **82**, 024016 (2010),
   [arXiv:1004.5098](https://arxiv.org/abs/1004.5098),
   [doi:10.1103/PhysRevD.82.024016](https://doi.org/10.1103/PhysRevD.82.024016).
4. L. Paulucci, E. J. Ferrer, J. E. Horvath, and V. de la Incera,
   "Bag vs. NJL models for color-flavor-locked strange quark matter,"
   *Journal of Physics G* **40**, 125202 (2013),
   [arXiv:1307.1504](https://arxiv.org/abs/1307.1504),
   [doi:10.1088/0954-3899/40/12/125202](https://doi.org/10.1088/0954-3899/40/12/125202).
