# Controlled compact-star EoS comparison

This thesis code solves the TOV and tidal equations and studies whether fixed
equation-of-state model families can be distinguished from their observable
curves. The repository retains the original controlled APR-1/CFL4 experiment
and adds an audited, leakage-resistant multi-family pilot.

The current experiment is deliberately narrow:

- hadronic class: the repository's analytic `APR-1` surrogate only;
- quark class: the published CFL4 benchmark with
  \(B=60\;\mathrm{MeV/fm^3}\), \(\Delta=100\;\mathrm{MeV}\), and
  \(m_s=150\;\mathrm{MeV}\);
- both classes: the same Gaussian sound-speed deformation with
  \(\epsilon_0=220\;\mathrm{MeV/fm^3}\),
  \(\sigma=50\;\mathrm{MeV/fm^3}\), and a deterministic sweep over \(A\).

This supports an **APR-1-surrogate versus fixed-CFL4 model-pair comparison**.
It does not, by itself, identify a general hadronic-versus-quark signature.
See [the controlled sweep rationale](docs/CONTROLLED_EOS_SWEEP.md) and
[the classification risk audit](docs/CLASSIFICATION_RISK_AUDIT.md).

The multi-family pilot expands this narrow pair to 9 hadronic and 9 published
CFL baselines, while preserving the same shared deformation controls. Its claim
is still limited to **repository hadronic-surrogate versus analytic CFL MIT-bag
model discrimination on unseen fixed EoS families**. See the
[family profile](docs/FAMILY_PILOT_PROFILE.md) and
[final classification report](docs/FAMILY_CLASSIFICATION_FINAL_REPORT.md).

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Generate the default 15 paired amplitudes (30 EoS curves):

```bash
python physics_main.py --force-regenerate
```

For a quick readiness run in a separate profile:

```bash
python physics_main.py --smoke-test --force-regenerate --data-root data/smoke
```

Build and audit clean tensors, optimize on training groups only, and perform the
single final model evaluation:

```bash
python main.py
```

Advanced utilities that repeatedly inspect held-out test labels are locked by
default. Unlock them only for a declared final analysis:

```bash
python main.py --skip-hpo --run-test-diagnostics
```

Run the noisy-data experiment after the clean split manifest exists:

```bash
python perturb_main.py
```

More options and artifact details are in
[the execution guide](docs/EXECUTION_GUIDE.md).

## Multi-family pilot

The family workflow deliberately uses separate runtime artifacts under
`data/family_pilot_v1/`:

```bash
python family_physics_main.py --data-root data/family_pilot_v1 --n-jobs 4 --force-regenerate
python family_ml_prepare.py --data-root data/family_pilot_v1
python family_shortcut_audit.py --data-root data/family_pilot_v1 --output-dir docs
python family_model_select.py --data-root data/family_pilot_v1 --output-dir docs
python family_development_robustness.py --data-root data/family_pilot_v1 --output-dir docs
```

The model profile in `framework/family_model_profile.json` was committed before
the two test families were opened. `family_final_test.py` enforces a one-shot
marker and refuses a repeat evaluation. Post-test interpretation and the strict
2.08-M_sun development sensitivity are produced by:

```bash
python family_posttest_report.py --data-root data/family_pilot_v1 --output-dir docs
```

The frozen radius-only logistic model achieved 1.00 family-balanced accuracy
on 13 training OOF families, 2 validation families, and the one-time 2-family
test. These are only 17 independent physical-family groups; the six A variants
per EoS are correlated sensitivity variants. A single low-mass radius already
separates much of this catalog, so the result must not be described as a
universal or opaque phase classifier.

## Data-integrity controls

- paired hadronic/quark curves share a `Sweep_ID` and never cross splits;
- train, validation, and test use contiguous blocks of \(A\);
- every curve is interpolated onto the same 21-point mass grid;
- clean and noisy variants reuse identical latent rows and split assignments;
- outer scalers are fitted on training rows only, while HPO fits a fresh scaler
  inside each inner-fit fold;
- microphysics and provenance columns are excluded from model features;
- hyperparameter search uses class-valid grouped inner cross-validation on
  training groups;
- validation labels choose decision thresholds; test labels do not.

For the family pilot, the split unit is `Family_Group_ID`, each complete curve
is one ML sample, all A variants remain together, every family has equal model
weight within its class, and the final test identities were locked before model
fitting. Shortcut probes demonstrate that raw sequence geometry and provenance
metadata are perfect label proxies; they are hard-forbidden from model input.

Unresolved scientific limitations, including baseline confounding and the need
for an external-baseline test, are recorded in the risk audit.
