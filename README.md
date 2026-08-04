# Controlled compact-star EoS comparison

This thesis code solves the TOV and tidal equations for a controlled pair of
equation-of-state models and studies whether their observable curves can be
distinguished by XGBoost and PyTorch MLP classifiers.

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

Unresolved scientific limitations, including baseline confounding and the need
for an external-baseline test, are recorded in the risk audit.
