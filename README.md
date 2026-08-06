# EoS Lab: controlled compact-star experiments

This repository solves the Tolman–Oppenheimer–Volkoff (TOV) and tidal equations
for controlled equation-of-state (EoS) studies. The supported post-thesis
interface is `eoslab.py`; experiment settings live in commented TOML files, so
scientific parameters do not need to be edited in source code.

The reference sensitivity experiment compares exactly two fixed parent models:

- the repository's analytic APR-1 hadronic surrogate;
- the CFL4 MIT-bag benchmark, with bag constant
  \(B=60\;\mathrm{MeV\,fm^{-3}}\), pairing gap
  \(\Delta=100\;\mathrm{MeV}\), and strange-quark mass
  \(m_s=150\;\mathrm{MeV}\).

Both use the same additive Gaussian sound-speed deformation, centred at energy
density \(\epsilon_0=220\;\mathrm{MeV\,fm^{-3}}\), with width
\(\sigma=50\;\mathrm{MeV\,fm^{-3}}\), while the dimensionless amplitude \(A\)
is swept. This is an APR-1-surrogate versus fixed-CFL4 sensitivity comparison.
It is not a universal matter-phase classifier and does not estimate an
astrophysical probability of quark matter.

## Quick start

Python 3.11 through 3.13 is supported. On Windows PowerShell:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python eoslab.py doctor
python eoslab.py list-eos
python eoslab.py validate configs/apr1_cfl4_reproduction.toml
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python eoslab.py doctor
python eoslab.py list-eos
python eoslab.py validate configs/apr1_cfl4_reproduction.toml
```

`requirements.txt` preserves the complete historical installation. Smaller
environments can use `requirements-physics.txt`, `requirements-ml.txt`, or
`requirements-dev.txt`; see [Configuration and installation](docs/CONFIGURATION.md).

## Supported commands

Validate before every run. Validation prints the resolved EoSs and provenance,
all parameter values and units, the exact amplitude grid, permitted amplitude
intervals, physical acceptance requirements, expected curve count, output
location, and scientific interpretation boundary.

```powershell
python eoslab.py doctor
python eoslab.py list-eos
python eoslab.py validate configs/apr1_cfl4_reproduction.toml
python eoslab.py run configs/apr1_cfl4_reproduction.toml
python eoslab.py status <run-directory>
python eoslab.py export-summary <run-directory>
python eoslab.py family develop configs/family_classification.toml
python eoslab.py family status
```

The four supplied profiles are:

- `configs/apr1_cfl4_reproduction.toml`: locked APR-1/CFL4 reference values;
- `configs/apr1_cfl4_exploration.toml`: editable one-pair sensitivity study;
- `configs/family_classification.toml`: audited development-only family workflow;
- `configs/smoke.toml`: a three-amplitude readiness and rejection-path check.

Unknown configuration fields, misspellings, an amplitude grid without \(A=0\),
or values outside the permitted interval stop validation. Runtime overrides are
recorded in the run manifest.

## Strict EoS and stable-branch requirements

The stored hadronic table includes the complete crust domain used by the TOV
solver, followed by the causal core. This exposed a small downward
energy-density jump at an internal boundary of the repository's legacy crust
fit. The new monotonicity check therefore rejects APR-1 at EoS validation and
retains both APR-1 and CFL4 tables as diagnostic, non-accepted records. It does
not smooth, clip, or silently repair the boundary.

A maximum mass is accepted only when the stellar sequence contains a resolved
first change from increasing to decreasing mass as central pressure rises, with
at least one valid model beyond the turning point. This is a turning-point
stability estimate, not a radial-oscillation calculation. A largest sampled
mass at the end of a sequence is rejected rather than reported as a maximum.

An independent core/TOV diagnostic also found that the current APR-1 surrogate
reaches its causal cutoff before a post-peak decrease is resolved. Thus fixing
the crust fit alone would not justify relabelling the former endpoint mass as a
bracketed maximum. The strict smoke profile intentionally exercises these
fail-closed paths and does not produce a reporting-grade stellar result.

Use `status` and inspect `tables/rejections.csv`. `export-summary` accepts
artifact-complete terminal runs, including runs with scientific rejections or
failed convergence, so the compact failure record can be archived.

## Run records

Every run uses a unique, non-overwriting directory under the ignored `runs/`
tree:

```text
runs/<experiment-name>/<timestamp>-<configuration-hash>/
  resolved_config.toml
  run_manifest.json
  logs/pipeline.log
  data/eos_tables.parquet
  data/stellar_curves.parquet
  tables/eos_summary.csv
  tables/rejections.csv
  tables/convergence.csv
  plots/
  report.md
```

The raw Parquet data stay outside version control. For any artifact-complete
terminal run, `export-summary` copies only the resolved configuration, manifest,
compact CSV tables, report, and selected figures into `reports/`.

## Family-level classification

The family workflow treats one complete curve as one sample, uses the same 21
masses from \(1.0M_\odot\) to \(2.0M_\odot\), and keeps every amplitude variant
of an EoS family in one partition. Preprocessing is fitted on training families
only. The reporting models are a dummy classifier and logistic regression;
XGBoost and the multilayer perceptron remain exploratory.

The existing final two-family test has already been opened once. The launcher
reports the saved result and integrity state, but refuses a second evaluation
and any evidence-writing development rerun:

```powershell
python eoslab.py family status
python eoslab.py family develop configs/family_classification.toml
```

The second command is expected to fail closed in the current frozen repository;
a future development experiment must use a separately versioned record.

Scores are family-balanced repository-model results, not physical probabilities
or general phase-detection performance. See the
[classification risk audit](docs/CLASSIFICATION_RISK_AUDIT.md) and
[final classification report](docs/FAMILY_CLASSIFICATION_FINAL_REPORT.md).

## Legacy entry points

`physics_main.py`, `main.py`, `perturb_main.py`, the `family_*.py` scripts, and
the Streamlit applications are retained to reproduce earlier thesis workflows
and artifacts. They are compatibility entry points, not the recommended way to
start a new controlled run. They may expose older parameter controls, output
layouts, classification wording, or test-diagnostic behaviour. Use `eoslab.py`
for new post-thesis work and consult [Supported workflows](docs/WORKFLOWS.md)
before invoking a legacy script.

Further scientific provenance and limitations are documented in the
[controlled sweep rationale](docs/CONTROLLED_EOS_SWEEP.md),
[EoS feasibility audit](docs/EOS_FEASIBILITY_AUDIT.md), and
[classification risk audit](docs/CLASSIFICATION_RISK_AUDIT.md).
