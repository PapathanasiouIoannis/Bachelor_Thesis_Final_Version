# Locked family pilot profile

This profile is the one-week classification dataset. It expands the controlled
APR-1/CFL4 experiment to multiple fixed EoS baselines without changing the
deformation mechanism.

## Scope and claim boundary

The experiment distinguishes repository hadronic surrogates from the analytic
CFL MIT-bag model on fixed EoS baselines that are absent from training. It does
not establish universal hadronic-versus-quark discrimination, because the
hadronic formula coefficients are currently traceable to a secondary thesis
rather than verified directly against every primary EoS source.

The locked profile is machine-readable in
`framework/family_pilot_profile.json`. It fixes:

- Gaussian center `epsilon0 = 220 MeV/fm^3`;
- Gaussian width `sigma = 50 MeV/fm^3`;
- the common amplitude grid `A = [0, 0.01, 0.02, 0.03, 0.04, 0.05]`;
- 9 hadronic and 9 CFL baselines;
- `M_max` in `[2.0, 3.0] M_sun`, `R_1.4` in `[9.5, 14.5] km`, and complete
  observable support from `1.0` to `2.0 M_sun`.

`MDI-3` is excluded because the amplitude stress test breaches the shared
radius screen for `A >= 0.03`; `MDI-2` preserves the same conservative MDI
family group. `PS` is excluded because the repository surrogate lacks a
primary citation.

## Generation and acceptance result

Run the production-resolution build with:

```powershell
py family_physics_main.py --data-root data/family_pilot_v1 --n-jobs 4 --force-regenerate
```

The accepted build contains 108 curves and 5,878 raw TOV sequence rows:

| Property | Result |
|---|---:|
| Hadronic curves | 54 |
| CFL curves | 54 |
| Fixed EoS baselines | 18 |
| Leakage-control family groups | 17 |
| Variants per EoS | 6 |
| Missing cells | 0 |
| Duplicate rows | 0 |
| Duplicate masses within a curve | 0 |

The generated dataset and manifest are stored under `data/family_pilot_v1/`
and remain untracked runtime artifacts. The checked-in amplitude evidence is
`docs/eos_family_amplitude_scan.csv`; its visualization is
`docs/EOS_FAMILY_AMPLITUDE_VIABILITY.png`.

## Required classification split

All six amplitude variants of an EoS must stay together. Related variants that
share `Family_Group_ID` must also stay together, notably `Ska` and `SkI4`.
Model selection may use only training and validation family groups. Test-family
labels remain locked until the final selected pipeline is frozen. A secondary
CFL robustness check should hold out `Parameter_Block_ID`, since all CFL curves
still share one analytic model superfamily.
