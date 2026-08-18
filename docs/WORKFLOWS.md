# Supported workflows

`eoslab.py` is the supported entry point for post-thesis work. It separates
configuration validation, scientific execution, artifact inspection, compact
export, and audited family development.

## 1. Readiness and discovery

```powershell
python eoslab.py doctor
python eoslab.py list-eos
```

`doctor` checks the supported Python version, required imports, analytic
hadronic library, and repository output access. `list-eos` shows the available
hadronic repository surrogates and published analytic CFL MIT-bag tuples with
their parameters and provenance status.

## 2. Pair-sensitivity experiment

Start by validating the chosen profile:

```powershell
python eoslab.py validate configs/apr1_cfl4_reproduction.toml
python eoslab.py run configs/apr1_cfl4_reproduction.toml
```

For each amplitude \(A\), the runner generates one hadronic EoS and one quark
EoS with a shared sweep identifier. A pair is accepted only if both members
pass. Complete pressure, energy-density, and squared-sound-speed tables are
retained, including the hadronic crust domain used by the TOV solver. Rejected
pairs remain in the EoS table with validation and pair-acceptance fields; they
are excluded from accepted stellar summaries.

The runner requires finite, strictly increasing pressure and energy density and
\(0<c_s^2\leq1\). It checks APR-1 crust/core matching, CFL surface stability and
pairing conditions, and recovery of the undeformed \(A=0\) baseline. The
maximum pointwise relative pressure-recovery tolerance is \(2\times10^{-4}\).
Pressure reconstruction uses cumulative Simpson quadrature in the framework;
this numerical refinement is covered by the undeformed-recovery test.

The causal-prefix cutoff is explicit. The table and manifest record the causal
cutoff pressure and energy density, discarded suffix count, and first discarded
value of \(c_s^2\). Values are never clipped or silently repaired.

### Stable stellar branch

Stellar models are ordered by increasing central pressure. The retained branch
ends at the first resolved change from increasing to decreasing mass and must
include at least one valid point beyond that maximum. A sequence whose largest
mass is its last sampled point is rejected. This is a turning-point stability
estimate, not a full radial-oscillation calculation.

Complete-table validation currently finds a small downward energy-density jump
at an internal boundary of the repository's legacy hadronic crust fit. APR-1 is
therefore rejected before stellar reporting; the runner retains the APR-1 and
CFL4 EoS tables and their two EoS-shape plot groups as diagnostics. Separately,
a core/TOV diagnostic found that APR-1 reaches its causal EoS boundary before a
post-peak mass decrease is resolved. Both findings are scientific blockers,
not conditions the launcher repairs or bypasses.

Inspect any run with:

```powershell
python eoslab.py status runs/<experiment-name>/<timestamp>-<configuration-hash>
```

### Convergence and plots

Production profiles request checks at the minimum, zero, and maximum amplitude
using doubled EoS resolution, doubled central-pressure resolution, and
ten-times tighter TOV tolerances. Agreement limits are
\(0.01M_\odot\) for maximum mass, 0.05 km for radius at
\(1.4M_\odot\), and 2% for tidal deformability at \(1.4M_\odot\).

An EoS-valid but stellar-rejected run produces the first two diagnostic plot
groups. A fully accepted run produces all four:

1. squared sound speed against energy density;
2. pressure against energy density;
3. stable mass–radius and tidal curves; and
4. maximum mass, radius at \(1.4M_\odot\), and tidal deformability at
   \(1.4M_\odot\) against amplitude.

### Compact export

Raw run directories are ignored by Git. Export any artifact-complete terminal
run (`completed`, `completed_with_rejections`, or `failed_convergence`):

```powershell
python eoslab.py export-summary <run-directory>
```

The command copies the resolved configuration, manifest, compact summary,
rejection and convergence tables, Markdown report, and selected PNG figures to
`reports/<experiment-name>/<run-name>/`. It does not copy the full EoS or
stellar-curve Parquet data.

## 3. Family-classification development

```powershell
python eoslab.py family status
python eoslab.py validate configs/family_classification.toml
python eoslab.py family develop configs/family_classification.toml
```

Before a final-test opening, the development command runs generation, curve
preparation, shortcut audit, low-capacity model selection, and family-label
robustness checks. It never invokes the final-test script. In this repository
the final test has already been opened, so the command now refuses every
evidence-writing development rerun. A future experiment must use separately
versioned data and report directories.

Safeguards include:

- one complete stellar curve per sample on the same 21 masses from
  \(1.0M_\odot\) to \(2.0M_\odot\);
- physical-family splitting, with every amplitude variant kept together;
- family and EoS disjointness checks;
- preprocessing fitted using training families only;
- exclusion of metadata and provenance fields from model input;
- amplitude, row-count, mass-support, single-mass radius, and single-mass tidal
  shortcut probes;
- family-level label-permutation checks;
- equal family weighting and exact per-family reporting; and
- final-test access-history and integrity checks.

The primary reporting models are a dummy classifier and logistic regression.
Random forest, XGBoost, and the multilayer perceptron are exploratory and cannot
win reporting-grade selection. The available catalog has 17 configured
split-family groups, but only two broad model superfamilies: repository
hadronic surrogates and analytic CFL MIT-bag models. The final partition
contains two held-out family groups, so aggregate metrics must always be
accompanied by exact per-family results.

### Final-test lock

The final family test was opened once during the recorded thesis analysis. Its
marker, open count, saved result, and integrity hashes are now read-only. The
launcher reports them through `family status`, verifies the locked development
evidence hashes, and refuses another evaluation or any development rerun that
could overwrite the frozen evidence. Never
describe a classifier score as a physical probability of quark matter.

## 4. Legacy compatibility workflows

The following scripts remain available to reproduce historical artifacts:

- `physics_main.py` for the earlier controlled physics pipeline;
- `main.py` and `perturb_main.py` for the earlier clean/noisy model pipelines;
- individual `family_*.py` scripts used to build the recorded family pilot;
- `app_ref.py` and `perturb_app_ref.py` for the historical Streamlit views.

They do not share all guarantees of `eoslab.py`: some use older output trees,
parameter locations, classification language, or test-diagnostic controls.
They should not be mixed into a new managed run directory. Treat them as
reproduction and compatibility entry points, and use the unified launcher for
new experiments. Their datasets and trained-model artifacts are generated
locally rather than shipped in Git; see [Runtime artifacts and locked
evidence](ARTIFACT_POLICY.md) before invoking them from a clean checkout.
