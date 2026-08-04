# Pipeline execution guide

Use a distinct `--data-root` for each experiment. This prevents an old physics
dataset or split manifest from being mistaken for a new configuration.

## 1. Generate the controlled physics dataset

```bash
python physics_main.py --force-regenerate --data-root data/apr1_cfl4
```

Defaults:

- one repository `APR-1` surrogate baseline;
- one fixed CFL4 tuple: \(B=60\;\mathrm{MeV/fm^3}\),
  \(\Delta=100\;\mathrm{MeV}\), \(m_s=150\;\mathrm{MeV}\);
- \(\epsilon_0=220\;\mathrm{MeV/fm^3}\) and
  \(\sigma=50\;\mathrm{MeV/fm^3}\);
- 15 common amplitudes from \(A=-0.05\) through \(A=0.09\), including
  the \(A=0\) control.

Only the A support and number of points are configurable from the command line:

```bash
python physics_main.py --a-min -0.05 --a-max 0.09 --a-points 15 \
  --curves-per-batch 5 --n-jobs 4 --force-regenerate \
  --data-root data/apr1_cfl4
```

The generator rejects an A outside either model's causal/stable interval. It
also aborts rather than promoting an incomplete pair if a TOV curve fails the
common viability cuts. The run manifest records the exact amplitudes, fixed
parameters, admissible intervals, configuration hash, source hash, and batch
results.

## 2. Build clean tensors and train

```bash
python main.py --data-root data/apr1_cfl4
```

The data stage interpolates every EoS onto the same 1.0--2.0 \(M_\odot\) mass
grid (21 rows per curve), creates paired contiguous-A splits, saves aligned row
and curve provenance, and fits the outer scaler on training rows only.
Hyperparameter optimization reads only an unscaled training artifact, uses
grouped inner folds, and fits preprocessing independently on each inner-fit
partition.

`main.py` performs the final model evaluation but does not automatically run
the large suite of test-label diagnostics. For a declared final diagnostic run:

```bash
python main.py --skip-hpo --run-test-diagnostics \
  --data-root data/apr1_cfl4
```

Do not repeatedly use that command during model development.

## 3. Build the noisy variant

```bash
python perturb_main.py --data-root data/apr1_cfl4
```

The noisy pipeline reuses the clean experiment's latent `Row_ID`s and shared
split manifest. Independent random seeds are used for train, validation, and
test noise. The current Gaussian error law remains a sensitivity experiment,
not a validated NICER/LIGO likelihood model.

## 4. Audit isolation

```bash
python -m src.ml.audit_leakage --include-perturbed \
  --data-root data/apr1_cfl4
```

The audit checks approved feature columns, row-to-metadata alignment,
`Curve_ID`/`Sweep_ID`/group disjointness, contiguous A blocks, equal rows per
curve, train-only scaling, and clean/noisy split identity.

## Interpretation boundary

Results may be described as discrimination between the selected APR-1 surrogate
and fixed CFL4 benchmark. A phase-general claim requires a locked external test
with unseen hadronic baselines and unseen quark parameter tuples.
