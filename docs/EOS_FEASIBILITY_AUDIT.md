# EoS Literature and A=0 Feasibility Audit

## Outcome

The production framework built and sent all 40 catalog entries through the same undeformed `A=0` TOV path. Under the current strict project screens, **8/21 hadronic surrogates** and **6/19 published CFL tuples** pass numerically. Removing the uncited PS surrogate leaves **7 hadronic and 6 CFL baselines recommended for a strict pilot**.

Using a conventional `Mmax >= 2.0 Msun` diagnostic instead of the project's harder `2.08 Msun` point-estimate cut yields **10 provenance-screened hadronic and 9 CFL baselines**. This larger sensitivity set is the practical one-week option; the strict `2.08` result must still be reported beside it.

No classifier was trained. This is the family-availability gate that must precede dataset generation.

The numerically strict common causal/stable amplitude intersection is **(-0.041039, 0.525085]**. It excludes the current lower endpoint solely because PS requires `A > -0.041039`. PS is also the one model rejected by the provenance screen. The recommended strict intersection is **(-0.082951, 0.525085]**, which does contain the current controlled sweep `[-0.05, 0.09]`.

## Provenance finding

The 21 repository hadronic expressions exactly match the formulas listed in Stergakis (2025), but the exact coefficients, fit domain, and fit uncertainty have not been verified in the cited primary model papers. Every hadronic row therefore has `exact_formula_primary_verified=false`. They are suitable only as transparently labelled **repository surrogates** until compared against primary/tabulated EoSs. `PS` is weaker still: the fit source attaches no primary citation to it.

The CFL side has stronger provenance: all 19 tuples and their reference maximum-star values appear in Tables I-II of Vasquez Flores & Lugones (2017). The repository solver agrees within 3% in both `Mmax` and the radius at `Mmax` for **19/19** successfully checked tuples. Numerical deltas are in `eos_feasibility_results.csv`.

## Which fixed values are literature-backed?

- `B=60 MeV/fm^3`, `Delta=100 MeV`, `m_s=150 MeV` are the published CFL4 tuple.
- `Mmax=2.08 Msun` is the central measured mass of PSR J0740+6620, not a hard lower confidence bound. This audit therefore reports both the project `2.08` screen and a separate `2.0` diagnostic.
- `epsilon0=220 MeV/fm^3`, `sigma=50 MeV/fm^3`, and the sampled `A` values are project-defined deformation coordinates. They are not values inferred from APR or CFL literature.
- `R1.4 in [9.5, 14.5] km` is treated here as a deliberately broad common-support engineering screen, not as a single published posterior interval.

For the strict recommended CFL entries, the largest Gaussian weight actually reached on each EoS grid ranges from 0.945 to 1.000. The same numeric `(epsilon0, sigma)` therefore does not produce exactly the same effective perturbation strength across parameter tuples; this overlap value must travel with each generated family.

## Classification consequence

The hadronic catalog contains 13 conservative family groups before physics screening. The CFL table contains 19 fixed-tuple families in 5 bag-constant blocks, but every tuple shares the same analytic CFL MIT-bag theory. The strict recommended set retains 6 hadronic family groups, 6 CFL tuple families, and only 2 CFL bag-constant blocks. The `2.0 Msun` sensitivity set raises these to 8, 9, and 3, respectively.

A primary family-held-out pilot can therefore hold out complete fixed EoS tuples and all their `A` variants. A harsher secondary check may hold out a complete CFL bag-constant block. Neither test establishes generalization to NJL, perturbative-QCD, or other quark-matter theories.

Proceed to dataset construction only with accepted rows, retain `family_group_id`, and keep the hadronic surrogate warning in every manifest. If the strict eligible counts are too small for the locked split, the scientifically honest fast fallback is a limited multi-baseline pilot with a narrower claim—not row-wise splitting of the same curves.

## Detailed acceptance table

| Class | EoS | Family group | Numeric pass | Recommended | Mmax | R1.4 | A interval | Reason |
|---|---|---|---|---|---|---|---|---|
| hadronic | APR-1 | H_APR | yes | yes | 2.289 | 12.984 | (-0.1088, 0.8792] | accepted |
| hadronic | BGP | H_BGP | yes | yes | 2.420 | 13.585 | (-0.1312, 0.8535] | accepted |
| hadronic | BL-1 | H_BL | yes | yes | 2.108 | 12.694 | (-0.1014, 0.8903] | accepted |
| hadronic | BL-2 | H_BL | no | no | 1.990 | 12.494 | (-0.0972, 0.8957] | does not cover common 1.0-2.0 Msun grid; Mmax below project screen 2.08 Msun |
| hadronic | DH | H_DH | no | no | 2.273 | 16.090 | (-0.1694, 0.8249] | R1.4 outside common 9.5-14.5 km screen |
| hadronic | HHJ-1 | H_HHJ | no | no | 1.985 | 12.909 | (-0.1095, 0.8822] | does not cover common 1.0-2.0 Msun grid; Mmax below project screen 2.08 Msun |
| hadronic | HHJ-2 | H_HHJ | yes | yes | 2.144 | 13.226 | (-0.1190, 0.8705] | accepted |
| hadronic | HLPS-2 | H_HLPS | no | no | 3.202 | 20.565 | (-0.2808, 0.6891] | Mmax above project upper screen 3.0 Msun; R1.4 outside common 9.5-14.5 km screen |
| hadronic | HLPS-3 | H_HLPS | no | no | 3.262 | 15.857 | (-0.2742, 0.4091] | Mmax above project upper screen 3.0 Msun; R1.4 outside common 9.5-14.5 km screen |
| hadronic | MDI-1 | H_MDI | no | no | 1.997 | 13.188 | (-0.1155, 0.8771] | does not cover common 1.0-2.0 Msun grid; Mmax below project screen 2.08 Msun |
| hadronic | MDI-2 | H_MDI | no | no | 2.004 | 13.424 | (-0.1208, 0.8723] | Mmax below project screen 2.08 Msun |
| hadronic | MDI-3 | H_MDI | no | no | 2.040 | 14.033 | (-0.1336, 0.8603] | Mmax below project screen 2.08 Msun |
| hadronic | MDI-4 | H_MDI | no | no | 2.067 | 14.759 | (-0.1449, 0.8500] | Mmax below project screen 2.08 Msun; R1.4 outside common 9.5-14.5 km screen |
| hadronic | NLD | H_NLD | no | no | 2.468 | 20.750 | (-0.1768, 0.8216] | R1.4 outside common 9.5-14.5 km screen |
| hadronic | PS | H_PS | yes | no | 2.472 | 12.324 | (-0.0410, 0.9438] | no underlying primary citation in fit source |
| hadronic | SCVBB | H_SCVBB | no | no | 2.014 | 12.290 | (-0.0916, 0.9010] | Mmax below project screen 2.08 Msun |
| hadronic | SkI4 | H_SKYRME | yes | yes | 2.276 | 13.748 | (-0.1369, 0.8515] | accepted |
| hadronic | Ska | H_SKYRME | yes | yes | 2.237 | 13.451 | (-0.1235, 0.8660] | accepted |
| hadronic | W | H_WALECKA | no | no | 2.653 | 14.927 | (-0.2036, 0.7726] | R1.4 outside common 9.5-14.5 km screen |
| hadronic | WFF-1 | H_WFF | no | no | 1.999 | 11.543 | (-0.0708, 0.9226] | does not cover common 1.0-2.0 Msun grid; Mmax below project screen 2.08 Msun |
| hadronic | WFF-2 | H_WFF | yes | yes | 2.094 | 12.057 | (-0.0830, 0.9088] | accepted |
| quark | CFL1 | Q_CFL1 | no | no | 2.067 | 11.076 | (-0.3504, 0.6558] | Mmax below project screen 2.08 Msun |
| quark | CFL10 | Q_CFL10 | yes | yes | 2.219 | 11.048 | (-0.4606, 0.5975] | accepted |
| quark | CFL11 | Q_CFL11 | no | no | 1.583 | 8.923 | (-61.8115, 117.4836] | does not cover common 1.0-2.0 Msun grid; Mmax below project screen 2.08 Msun; R1.4 outside common 9.5-14.5 km screen |
| quark | CFL12 | Q_CFL12 | no | no | 1.768 | 9.570 | (-5.3243, 8.5713] | does not cover common 1.0-2.0 Msun grid; Mmax below project screen 2.08 Msun |
| quark | CFL13 | Q_CFL13 | no | no | 1.629 | 9.101 | (-28.4200, 51.9126] | does not cover common 1.0-2.0 Msun grid; Mmax below project screen 2.08 Msun; R1.4 outside common 9.5-14.5 km screen |
| quark | CFL14 | Q_CFL14 | no | no | 2.070 | 10.360 | (-1.0391, 1.2293] | Mmax below project screen 2.08 Msun |
| quark | CFL15 | Q_CFL15 | no | no | 1.937 | 10.042 | (-1.7031, 2.3224] | does not cover common 1.0-2.0 Msun grid; Mmax below project screen 2.08 Msun |
| quark | CFL16 | Q_CFL16 | no | no | 1.594 | 8.787 | (-420.1199, 690.1364] | does not cover common 1.0-2.0 Msun grid; Mmax below project screen 2.08 Msun; R1.4 outside common 9.5-14.5 km screen |
| quark | CFL17 | Q_CFL17 | no | no | 1.848 | 9.560 | (-11.9811, 14.9002] | does not cover common 1.0-2.0 Msun grid; Mmax below project screen 2.08 Msun |
| quark | CFL18 | Q_CFL18 | no | no | 1.735 | 9.252 | (-40.1917, 56.8352] | does not cover common 1.0-2.0 Msun grid; Mmax below project screen 2.08 Msun; R1.4 outside common 9.5-14.5 km screen |
| quark | CFL19 | Q_CFL19 | no | no | 1.680 | 8.898 | (-789.5283, 1020.4511] | does not cover common 1.0-2.0 Msun grid; Mmax below project screen 2.08 Msun; R1.4 outside common 9.5-14.5 km screen |
| quark | CFL2 | Q_CFL2 | no | no | 1.844 | 10.371 | (-0.4419, 0.9546] | does not cover common 1.0-2.0 Msun grid; Mmax below project screen 2.08 Msun |
| quark | CFL3 | Q_CFL3 | yes | yes | 2.375 | 11.868 | (-0.3935, 0.6063] | accepted |
| quark | CFL4 | Q_CFL4 | yes | yes | 2.143 | 11.290 | (-0.3598, 0.6402] | accepted |
| quark | CFL5 | Q_CFL5 | yes | yes | 2.863 | 12.813 | (-0.4663, 0.5329] | accepted |
| quark | CFL6 | Q_CFL6 | yes | yes | 2.651 | 12.443 | (-0.4340, 0.5655] | accepted |
| quark | CFL7 | Q_CFL7 | no | no | 2.009 | 10.540 | (-0.5634, 0.8822] | Mmax below project screen 2.08 Msun |
| quark | CFL8 | Q_CFL8 | no | no | 1.835 | 10.039 | (-0.9977, 1.8022] | does not cover common 1.0-2.0 Msun grid; Mmax below project screen 2.08 Msun |
| quark | CFL9 | Q_CFL9 | yes | yes | 2.383 | 11.386 | (-0.4750, 0.5251] | accepted |

## Reproducible artifacts

- `eos_literature_catalog.csv`: source, family grouping, parameters, and provenance layer for all 40 entries.
- `eos_feasibility_results.csv`: complete numeric audit and rejection reasons.
- `eos_feasibility_summary.json`: counts, intersections, and fixed audit controls.
- `eos_feasibility_curves.npz`: the plotted undeformed stable branches.
- `EOS_FEASIBILITY_MR.png`: all undeformed M-R sequences; green is strict and provenance-recommended.
- `EOS_AMPLITUDE_INTERVALS.png`: model-specific causal/stable A support and the current requested sweep.

Run again with `py -m framework.audit_eos_feasibility --jobs 4`.

## Primary sources and exact-fit source

- Stergakis, *Reconstruction of the Equations of State (EoSs) of Compact Stars using machine and deep learning regression techniques* (2025): https://arxiv.org/abs/2509.13037
- Vasquez Flores & Lugones, *Constraining color flavor locked strange stars in the gravitational wave era* (2017): https://arxiv.org/abs/1702.02081
- Lugones & Horvath, *Color-flavor locked strange matter* (2002): https://arxiv.org/abs/hep-ph/0211070
- Fonseca et al., *Refined Mass and Geometric Measurements of the High-Mass PSR J0740+6620* (2021): https://arxiv.org/abs/2104.00880
