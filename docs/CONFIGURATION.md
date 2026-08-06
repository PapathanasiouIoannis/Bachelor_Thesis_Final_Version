# Configuration and installation

The supported launcher reads a strict TOML configuration. Every user-controlled
quantity has a descriptive stored name, and every dimensional quantity states
its unit. Editing Python source is neither required nor supported as a way to
change an experiment.

## Installation profiles

Python 3.11 through 3.13 is supported.

| File | Purpose |
|---|---|
| `requirements.txt` | Complete compatibility environment for all historical workflows |
| `requirements-physics.txt` | EoS generation, stellar solutions, reports, unified CLI, and low-capacity safeguards |
| `requirements-ml.txt` | Physics stack plus exploratory ML and legacy Streamlit applications |
| `requirements-dev.txt` | Physics stack plus the regression-test and lint tools |

Use the complete environment unless disk space or installation time is a
constraint:

```powershell
python -m pip install -r requirements.txt
```

The run manifest records the interpreter, platform, selected package versions,
Git revision, source-tree hash, resolved configuration, and any supported
runtime override. Dependency ranges support installation across the declared
Python versions; the manifest is the exact environment record for a run.

## Supplied profiles

| Profile | Mode | Intended use |
|---|---|---|
| `apr1_cfl4_reproduction.toml` | `reproduction` | Locked APR-1/CFL4 reference experiment |
| `apr1_cfl4_exploration.toml` | `exploration` | Editable sensitivity experiment with one parent EoS per matter type |
| `family_classification.toml` | `development` | Audited family-level classification development; final test is read-only |
| `smoke.toml` | `exploration` | Three-amplitude readiness and rejection-path check |

Reproduction mode accepts only APR-1, the CFL4 tuple
\(B=60\;\mathrm{MeV\,fm^{-3}}\), \(\Delta=100\;\mathrm{MeV}\),
\(m_s=150\;\mathrm{MeV}\), and the documented Gaussian deformation. Use the
exploration profile when changing one baseline, one quark tuple, or deformation
settings.

## Pair-sensitivity schema

The following names are the public configuration interface:

| TOML field | Meaning | Unit |
|---|---|---|
| `hadronic_eos.baseline` | One hadronic baseline identifier | none |
| `quark_eos.bag_constant_mev_fm3` | Bag constant \(B\) | \(\mathrm{MeV\,fm^{-3}}\) |
| `quark_eos.pairing_gap_mev` | Pairing gap \(\Delta\) | \(\mathrm{MeV}\) |
| `quark_eos.strange_quark_mass_mev` | Strange-quark mass \(m_s\) | \(\mathrm{MeV}\) |
| `deformation.center_energy_density_mev_fm3` | Gaussian centre \(\epsilon_0\) | \(\mathrm{MeV\,fm^{-3}}\) |
| `deformation.width_mev_fm3` | Gaussian width \(\sigma\) | \(\mathrm{MeV\,fm^{-3}}\) |
| `deformation.amplitude_start` | First amplitude \(A\) | dimensionless |
| `deformation.amplitude_stop` | Last amplitude \(A\) | dimensionless |
| `deformation.amplitude_step` | Amplitude increment | dimensionless |
| `physical_requirements.minimum_maximum_mass_msun` | Lower accepted maximum mass | \(M_\odot\) |
| `physical_requirements.maximum_maximum_mass_msun` | Upper accepted maximum mass | \(M_\odot\) |
| `physical_requirements.radius_1p4_min_km` | Lower radius at \(1.4M_\odot\) | km |
| `physical_requirements.radius_1p4_max_km` | Upper radius at \(1.4M_\odot\) | km |
| `execution.parallel_jobs` | Worker-process count | none |
| `execution.amplitudes_per_batch` | Planned batch size | amplitudes |

The parser rejects unknown sections, unknown fields, missing fields, incorrect
types, invalid profile combinations, and misspellings. The amplitude grid is
constructed exactly from decimal start, stop, and step values. It must include
\(A=0\); points are never silently shifted to make that happen.

## Validation

Run validation after every edit:

```powershell
python eoslab.py validate configs/apr1_cfl4_exploration.toml
```

Validation displays:

- the selected EoSs and their repository provenance;
- \(B\), \(\Delta\), \(m_s\), \(\epsilon_0\), and \(\sigma\), with units;
- every amplitude value \(A\);
- the permitted interval for each baseline and their common interval;
- the expected number of generated curves;
- mass and radius acceptance requirements;
- the unique output-directory pattern; and
- the permitted scientific interpretation.

The only command-line science-adjacent override is `--jobs`, which changes
parallel execution without changing the EoS. It is written to the manifest.
`--runs-root` changes only the location of the isolated run directory.

The named numerical presets resolve to explicit settings and are recorded in
the manifest and report:

| Preset | EoS grid points | Central-pressure points | TOV relative tolerance | TOV absolute tolerance |
|---|---:|---:|---:|---:|
| `production` | 10000 | 200 | `1e-8` | `1e-10` |
| `smoke` | 5000 | 80 | `1e-7` | `1e-9` |

`smoke` is a readiness and rejection-path check, not a reporting preset. The
locked reproduction profile requires `production` and the endpoint/zero
convergence checks.

Example actionable error:

```text
amplitude_stop = 0.20 exceeds the common permitted maximum of 0.09. Choose a smaller value.
```

## Stored and displayed names

Tables use explicit stored names while reports use plain scientific headings:

| Report heading | Stored column |
|---|---|
| Amplitude \(A\) | `deformation_amplitude` |
| Maximum mass \([M_\odot]\) | `maximum_mass_msun` |
| Radius at \(1.4M_\odot\) [km] | `radius_1p4_km` |
| Tidal deformability at \(1.4M_\odot\) | `tidal_deformability_1p4` |

Matter types are displayed as `hadronic` and `quark`, never as unexplained
numeric class labels.
