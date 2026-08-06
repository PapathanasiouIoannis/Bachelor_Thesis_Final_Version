# Deterministic APR-1-Surrogate Hadronic Sweep

## Scope

`src/physics/worker_hadronic_gen.py` is a thesis-era compatibility entry point
for the hadronic half of the controlled paired experiment. It no longer samples
randomly from the full hadronic library. Every requested sweep point uses only
the repository baseline named `APR-1`, and the common construction is delegated
to `framework/eos_sweep.py`.

For new managed runs, `eoslab.py` is the supported entry point. It reads the
selected TOML profile, validates the exact amplitude grid, and passes the
resolved baseline and deformation values explicitly through
`src/physics/experiment_runner.py`. The legacy worker instead reads mirrored
compatibility defaults from `src/config.py`; changing source code is not part
of the supported workflow.

The default controlled parameters are

$$
\epsilon_0=220\ \mathrm{MeV\,fm^{-3}},\qquad
\sigma=50\ \mathrm{MeV\,fm^{-3}},
$$

with the ordered amplitude grid

$$
A\in\{-0.05,-0.04,\ldots,0,\ldots,0.08,0.09\}.
$$

This gives 15 deterministic amplitudes at spacing $0.01$, including the
undeformed $A=0$ control. The Gaussian location, width, and amplitude support
are project-defined experimental controls, not parameters supplied by APR
literature. Their selection and the paired CFL4 construction are documented in
[Controlled APR-1-Surrogate / CFL4 EoS Sweep](../CONTROLLED_EOS_SWEEP.md).

## Repository APR-1 surrogate

The baseline implemented in `src/physics/get_eos_library.py` is

$$
\epsilon_{\mathrm{raw}}(P)
=0.000719964\,P^{1.85898}
+108.975\,P^{0.340074},
$$

where $P$ and $\epsilon$ are in $\mathrm{MeV\,fm^{-3}}$. Its baseline sound
speed is evaluated from the analytic derivative,

$$
c_{s,0}^2(P)
=\left(\frac{d\epsilon_{\mathrm{raw}}}{dP}\right)^{-1}.
$$

This expression must be described as the **repository APR-1 surrogate**. The
name APR-1 is not consistent across common sources:

- Koliogiannis and Moustakidis call A18+UIX `APR-1` and
  A18+$\delta v$+UIX* `APR-2`.
- Read et al. use `APR1` for a piecewise-polytropic fit, not the two-power
  expression above.
- The authoritative CompOSE entry called `APR` uses A18+$\delta v$+UIX*, not
  the interaction called APR-1 by Koliogiannis and Moustakidis.

The exact two-power coefficients are not supplied in the primary Akmal or Read
papers. A verbatim match has been found only in a later secondary thesis, which
does not independently establish the primary fitting procedure, fit error, or
validity interval. The generated data therefore must not be described as a
verified Read, CompOSE, or tabulated Akmal EoS.

Primary nomenclature sources include
[Akmal, Pandharipande, and Ravenhall (1998)](https://doi.org/10.1103/PhysRevC.58.1804),
[Read et al. (2009)](https://doi.org/10.1103/PhysRevD.79.124032), and
[Koliogiannis and Moustakidis (2020)](https://doi.org/10.1103/PhysRevC.101.015805).
The [CompOSE APR entry](https://compose.obspm.fr/eos/68) is an authoritative
database cross-check, not the source of the local analytic coefficients.

## Crust/core anchor and density shift

The low-pressure branch uses the repository's four analytic crust functions.
Their exact local coefficient set has not been independently established as
the published unified Douchin--Haensel SLy parameterization, so the controlled
path does not make that attribution.

The core is anchored at the configured pressure

$$
P_{\mathrm{trans}}=0.184\ \mathrm{MeV\,fm^{-3}}.
$$

At this pressure, `resolve_density_shifted_transition` computes

$$
\Delta\epsilon
=\epsilon_{\mathrm{raw}}(P_{\mathrm{trans}})
-\epsilon_{\mathrm{crust}}(P_{\mathrm{trans}}),
$$

and defines the matched core coordinate

$$
\epsilon_{\mathrm{core}}(P)
=\epsilon_{\mathrm{raw}}(P)-\Delta\epsilon.
$$

For the repository APR-1 surrogate,

$$
\epsilon_{\mathrm{raw}}(P_{\mathrm{trans}})\simeq61.27887,
\qquad
\epsilon_{\mathrm{crust}}(P_{\mathrm{trans}})\simeq46.53570,
$$

so $\Delta\epsilon\simeq14.74317\ \mathrm{MeV\,fm^{-3}}$. The shift makes
$\epsilon$ continuous at the fixed anchor and leaves
$d\epsilon_{\mathrm{raw}}/dP$ unchanged. It does not establish derivative or
chemical-potential continuity and should not be presented as a literature
crust/core phase-equilibrium construction.

The Gaussian is evaluated on this shifted core energy-density coordinate.
Thus the controlled value $\epsilon_0=220\ \mathrm{MeV\,fm^{-3}}$ refers to
the matched repository coordinate, not the raw two-power value.

## Gaussian sound-speed deformation

For each `SweepPoint`, the framework constructs

$$
g(\epsilon)=
\exp\!\left[-\frac{1}{2}
\left(\frac{\epsilon-\epsilon_0}{\sigma}\right)^2\right]
$$

and

$$
c_{s,A}^2(\epsilon)
=c_{s,0}^2(\epsilon)+A g(\epsilon).
$$

$A$ is dimensionless. Before any TOV sequence is generated, the worker derives
the baseline-specific amplitude interval satisfying

$$
0<c_{s,A}^2(\epsilon)\leq1
$$

over the retained baseline domain and verifies every requested amplitude
against it. Equivalently, because $g>0$ on its numerical support,

$$
A>
\max_\epsilon\left[-\frac{c_{s,0}^2(\epsilon)}{g(\epsilon)}\right],
\qquad
A\leq
\min_\epsilon\left[\frac{1-c_{s,0}^2(\epsilon)}{g(\epsilon)}\right].
$$

For the current shifted APR-1 grid, this interval is approximately

$$
-0.10875<A\leq0.87916,
$$

so the default $[-0.05,0.09]$ grid lies inside the hadronic causal and
thermodynamically stable support. The common experiment uses the more
restrictive validated support shared with CFL4; see the controlled-sweep note
linked above.

## Causal prefix and pressure reconstruction

The controlled framework does **not** clip sound speeds to a numerical floor or
cap. It finds the first non-finite, non-positive, or superluminal point and
retains the stable causal prefix. A deformation that leaves fewer than four
valid grid points fails. Because configured amplitudes are validated in
advance, the normal high-density endpoint is the baseline's first causal
crossing rather than a silently repaired part of the Gaussian.

After changing $c_s^2$, the original pressure relation cannot be retained. The
framework restores thermodynamic consistency by integrating

$$
P_A(\epsilon)
=P_{\mathrm{trans}}
+\int_{\epsilon_{\mathrm{trans}}}^{\epsilon}
c_{s,A}^2(u)\,du.
$$

It performs this integral with cumulative Simpson quadrature and creates
monotone PCHIP interpolators for $\epsilon_A(P)$ and $c_{s,A}^2(P)$. The
analytic crust remains active for $P\leq P_{\mathrm{trans}}$. Evaluation above
the retained causal pressure endpoint fails instead of extrapolating the core.
Managed runs verify the reconstruction through the maximum pointwise relative
pressure error of the undeformed $A=0$ control.

## Common stellar validation and pairing

Every reconstructed APR-1-surrogate EoS is passed to the same sequence
validation path as its CFL4 partner. The common controls require

- a non-empty stable TOV sequence that reaches the configured low-mass branch;
- sufficient physically valid points and finite canonical $1.4\,M_\odot$
  features;
- $2.08\leq M_{\max}/M_\odot\leq3.0$; and
- $9.5\leq R_{1.4}/\mathrm{km}\leq14.5$.

Failure of any requested amplitude aborts that controlled generation rather
than substituting another EoS. Accepted rows record the exact $A$,
$\epsilon_0$, $\sigma$, baseline name, and generation seed.

`SweepPoint.sweep_id` supplies the stable identifier `A00000`, `A00001`, and
so on. The APR-1-surrogate curve and fixed-CFL4 curve produced for the same
amplitude share this `Sweep_ID`; their class-specific `Curve_ID` values remain
distinct. Dataset validation requires every `Sweep_ID` to contain both labels
and verifies that both members store the same amplitude. This identifier is
also the grouping unit used to prevent paired members from crossing machine-
learning splits.
