# Classification Validity and Leakage Audit

## Scope and status

This document audits the data and machine-learning design for the controlled Gaussian-amplitude experiment. It distinguishes safeguards enforced in the working branch from remaining engineering gaps and scientific limitations that code changes alone cannot resolve.

The controlled experiment currently means:

- hadronic model: APR-1 only;
- quark model: one fixed CFL4 tuple, `B = 60 MeV/fm^3`, `Delta = 100 MeV`, and `m_s = 150 MeV`;
- shared Gaussian deformation: `epsilon0 = 220 MeV/fm^3`, `sigma = 50 MeV/fm^3`;
- shared sweep: 15 amplitudes from `A = -0.05` through `A = 0.09`;
- one APR-1 and one CFL4 EoS for every amplitude, linked by a common `Sweep_ID`.

The audit is read-only with respect to the measured legacy datasets. The measurements below describe `data/1K`; they are not results from the new controlled generator.

## Executive conclusion

The experiment can support a controlled **APR-1-versus-CFL4 model-pair discrimination** result. It cannot, by itself, support a general claim that a classifier distinguishes hadronic matter from quark matter.

With one baseline per class, class and baseline are perfectly confounded:

\[
\text{class label} \equiv \text{APR-1 or fixed CFL4 baseline}.
\]

Varying `A` supplies repeated, highly correlated observations within the same two baselines. It does not provide independent replication of the matter class. A shared `Sweep_ID` prevents paired-amplitude leakage but cannot remove this identifiability limit.

The current 15-value sweep also provides only 15 paired statistical groups. It is too small for a defensible train/validation/test partition plus model selection with high-capacity XGBoost or MLP models. A finer `A` grid would add interpolation density, not independent physical diversity.

## Controls implemented in the working branch

### Generation-layer controls

The following safeguards are implemented in the controlled generation path:

- `framework/eos_sweep.py` provides one deformation and pressure-reconstruction implementation for both classes.
- `src/config.py` fixes APR-1, the CFL4 tuple, `epsilon0`, `sigma`, and the declared `A` grid.
- `Sweep_ID` identifies the shared amplitude, while `Curve_ID` remains class- and baseline-specific.
- `physics_main.py` verifies that every `Sweep_ID` has both labels, both curves store the same `A`, only APR-1 appears in the hadronic class, and the quark parameters remain fixed.
- The requested amplitudes are checked against both baselines' causal and thermodynamically stable amplitude intervals before generation.
- `src/physics/controlled_generation.py` applies the same maximum-mass and `R_1.4` acceptance windows to both classes.
- Generation fails rather than silently replacing an invalid or missing sweep member.
- The generation manifest explicitly labels the experiment as `APR-1 versus fixed-CFL4 model-pair discrimination` and records the controlled physics parameters.

These controls address parameter drift, missing pairs, the former class-specific radius cut, and the former asymmetry in which only hadronic EoS received a Gaussian deformation.

### ML and data controls

The controlled ML path now enforces the following safeguards:

- `src/ml/dataset.py` interpolates every accepted curve onto the same 21 mass nodes from `1.0` through `2.0 M_sun` using shape-preserving interpolation. Construction fails if a curve cannot support the complete grid, and every retained curve must contribute exactly 21 rows.
- Preprocessing does not deduplicate observations globally. Duplicate mass coordinates may be collapsed only inside one `Curve_ID`, so a shared physical point cannot delete a row from another curve or class.
- Stable `Row_ID` values identify the common-grid rows. Each tensor artifact has a row-aligned sidecar containing `Row_ID`, `Curve_ID`, `Sweep_ID`, `Group_ID`, `A`, baseline, label, and split.
- `src/ml/splitting.py` keeps the hadronic and quark members of a `Sweep_ID` together and assigns paired, contiguous blocks in ordered `A`. It persists one shared split manifest and fails if an existing manifest has a different group/amplitude assignment.
- Clean and perturbed datasets start from the same latent rows and the same split manifest. Perturbations are applied only after split assignment, with independent train, validation, and test seeds.
- The final model boundary has an explicit allowlist: `Mass`, `Radius`, and `log10_Lambda`. Provenance, generator parameters, and identifiers remain in sidecars rather than model tensors.
- The outer `StandardScaler` is fitted only on training rows. Hyperparameter optimization reads a separate unscaled training artifact and fits a fresh scaler on each inner-fit fold, so an inner score fold cannot influence preprocessing. The leakage audit checks the feature schema, train-only scaling, row alignment, unique row identity, group disjointness, paired amplitudes, contiguous blocks, and common-grid row counts.
- Hyperparameter optimization loads the training artifact only. Paired controlled sweeps use contiguous grouped inner folds keyed by ordered `Group_ID`; legacy single-class curve groups use stratified grouped folds and reject single-class fit or score partitions. Validation and test artifacts are not inner-HPO folds.
- Advanced test diagnostics are locked behind an explicit command-line opt-in plus runtime authorization. They are excluded from the default workflow.
- Classification thresholds are selected from validation predictions. Final scripts use explicit, consistent names for trapezoidal PR-AUC, macro-F1, quark-class F1, Brier score, and the decision threshold.
- Run artifacts record source-tree, configuration, latent-dataset, post-transform tensor-input, split-manifest, individual train/validation/test tensor, scaler, sidecar, and unscaled-HPO artifact hashes, along with the noise transform, selected feature subset, approved feature order, and experiment scope. Each best-parameter file has a content-hashed lineage sidecar, and final training rejects any missing or mismatched component, source, configuration, tensor, transform, split, feature, or artifact hash.

These controls remove the previously observed unequal curve weighting, cross-curve deduplication, independently redrawn clean/noisy populations, direct feature leakage, randomly interleaved amplitudes, and accidental default execution of advanced test diagnostics. They do not make neighboring amplitudes independent, increase the number of physical baselines, or turn the 15 paired groups into a large statistical sample.

## Remaining engineering gaps

The following protections are not yet complete:

- Evaluation does not yet report curve- or `Sweep_ID`-clustered confidence intervals. Row-level metrics therefore must not be interpreted as having a row-level effective sample size.
- Predicted probabilities are not fitted-calibration outputs. Brier scores and reliability plots diagnose calibration but do not create it.
- The perturbation model remains one simple independent Gaussian law. It does not represent covariance, non-Gaussian posteriors, source-dependent uncertainty, selection effects, or repeated predeclared noise families.
- There is no locked external-baseline dataset. This is both an evaluation gap and the central limit on the scientific claim.

The grouped inner-HPO mechanism prevents direct test leakage, but with only 15 correlated `Sweep_ID` groups it remains statistically fragile. It is not a substitute for nested evaluation across independent physics families.

## Unresolved scientific limitations

### Model-pair confounding

No statistical split of the current data can separate a matter-class effect from a baseline effect. Holding out amplitudes tests interpolation or extrapolation in `A` for APR-1 and CFL4; it does not test unseen hadronic or quark physics.

Resolution requires a locked external dataset with hadronic baselines and quark parameter tuples absent from training and all model-selection decisions. Until then, all reports, plots, manifests, and application text must use model-pair language rather than phase-general language.

### Insufficient independent groups for nested blocked CV

There are 15 `Sweep_ID` groups. A conventional 80/10/10 split would leave only one or two amplitudes in validation and test. Five-fold nested CV would make inner validation estimates unstable and encourage selection on individual amplitudes.

Acceptable options are:

1. treat the 15-point run as a descriptive physics/separability study with predeclared simple analyses and no HPO; or
2. add genuinely independent physics families, then perform nested grouped evaluation across those families and blocked amplitude regions.

Adding many more closely spaced `A` values does not solve the independence problem.

### Synthetic-to-observational generalization

Performance on one Gaussian noise law is not evidence of deployment performance on NICER/LIGO posterior samples. Measurement errors may be correlated, non-Gaussian, source-dependent, and affected by selection. Calibration on a simulated 50/50 class construction is not an astrophysical posterior probability without an explicit target prior and external validation.

## Measured legacy-data findings

The following diagnostics were measured from `data/1K` on 2026-08-04.

| Finding | Measured result | Consequence |
|---|---:|---|
| Curves per class | 500 hadronic, 500 quark | Curve counts were balanced in this profile. |
| Rows per class | 16,351 hadronic, 29,188 quark | Row counts were not balanced despite equal curves. |
| Median rows per curve | 33 hadronic, 58 quark | Numerical solver sampling assigned different weight by class. |
| Clean test composition | 42 H / 58 Q curves; 1,296 H / 3,410 Q rows | Row-level metrics were dominated by quark grid points. |
| Global clean deduplication | 579 rows removed, all hadronic | Expected shared low-density points were treated as duplicates. |
| Curves affected by deduplication | 28 affected; 3 removed entirely | Curve identity and sweep coverage were altered before splitting. |
| Clean/noisy split agreement | 315 of 1,000 curves changed split | Clean-versus-noisy comparisons used different populations. |
| Common clean/noisy test curves | 17 curves | Test-set overlap was only 17 curves, preventing a paired comparison. |
| Simple logistic diagnostic | AP 0.935 from radius alone; AP 0.945 from mass and radius | Generator support and acceptance-cut ablations are mandatory. This is a warning signal, not proof of leakage. |

The unequal row counts originate from treating all stable TOV pressure-grid points as independent ML rows while `src/physics/solve_sequence.py` uses class-specific pressure grids. Class weighting corrects only the aggregate label imbalance; it does not remove within-curve dependence or arbitrary mass-density weighting.

These measurements document failure modes of the legacy `data/1K` workflow. The controlled pipeline's common-grid construction, absence of global deduplication, and shared split manifest are specifically designed to prevent their recurrence; the table is not a description of newly generated controlled artifacts.

## Severity-ranked unresolved risks

### Critical: baseline is identical to class

The classifier can learn APR-1 versus CFL4 and still fail on every unseen EoS. This is an experimental-design confound, not a software defect.

Required action: constrain the claim now and require external-baseline validation for any later phase-general claim.

### Critical: effective sample size remains 15 correlated groups

The common grid makes curve weight explicit and `Sweep_ID` grouping prevents the paired classes from crossing splits, but the 21 points on a curve are deterministic repeated measurements, not 21 independent observations. The 15 amplitudes also share the same two undeformed baselines and vary smoothly with `A`.

Required action: make `Sweep_ID` the unit of uncertainty, report per-curve or paired-group summaries, add clustered confidence intervals, and state `n = 15` rather than implying that the number of grid rows is the sample size.

### High: held-out contiguous amplitudes test edge extrapolation only

The former random-amplitude leakage is controlled: both labels remain paired and split membership is assigned in contiguous `A` blocks. In the present 15-point layout, however, validation and test estimates depend on very small edge blocks. They measure extrapolation along one constructed deformation axis, not generalization across EoS families.

Required action: predeclare the block assignment, report the exact held-out amplitudes, stratify results by `A`, and do not average away the distinction between interpolation, edge extrapolation, and external-baseline generalization.

### High: model selection is unstable despite grouped inner folds

The optimization scripts now use training-only grouped folds and do not use the locked validation or test partitions as inner HPO folds. This closes direct test leakage. It does not supply enough independent information for reliable selection of high-capacity XGBoost or MLP configurations: after the contiguous outer split, only a small number of highly related training groups remain.

Required action: for the 15-point experiment, predeclare a simple model or treat HPO findings as exploratory. A defensible generalization study needs nested grouped evaluation across additional independent physics families.

### High: synthetic-to-observational validation is absent

The current clean/noisy comparison is properly paired at the latent-row and split-manifest level, and train, validation, and test perturbations use distinct seeds. Nevertheless, one independent Gaussian error law is a narrow simulator assumption rather than an observational validation.

Required action: test predeclared alternative noise laws and covariance structures, retain every realization of a latent group in one split, and validate on representative posterior samples before making deployment claims.

### Medium: probabilities are not calibrated astrophysical posteriors

Metric names and validation-selected thresholds are now consistent, but no independent calibration fit or target astrophysical prior is applied. A Brier score or reliability diagram is diagnostic evidence, not a calibration procedure.

Required action: call uncalibrated outputs model scores, and require an independent calibration design, target prior, and external validation before using `P(quark | observation)` language.

### Medium: lineage checks cover managed training entrypoints, not every external consumer

The clean and perturbed final-training scripts fail closed when a best-parameter artifact or its source/configuration/tensor/split/feature lineage differs. A copied model consumed outside these managed entrypoints can still lose that context.

Required action: distribute model weights together with their manifests and treat a detached artifact as unauditable.

### Medium: authorized test access still requires governance

Advanced diagnostics no longer run by default and require explicit runtime authorization. Final evaluation necessarily reads the test set once, but repeated authorized reruns could still influence subsequent modeling choices through human feedback.

Required action: predeclare the final run, archive its configuration and hashes, log every test authorization, and regenerate a new locked external test set after any test-informed model change.

## Acceptance status before ML results are reported

### Enforced gates in the controlled path

Generation and scope:

- Every requested `Sweep_ID` must contain exactly one APR-1 curve and one curve from the fixed CFL4 tuple, with the same declared `A`.
- `epsilon0`, `sigma`, the amplitude list, admissible intervals, common physical filters, source revision, configuration hash, and model-pair scope are recorded or validated during generation.
- A missing, invalid, replaced, or unpaired amplitude causes failure instead of silent dataset shrinkage.

Dataset construction:

- Every accepted curve contributes exactly the configured 21 common mass nodes; raw central-pressure sampling density does not reach model training or evaluation.
- Deduplication does not cross a `Curve_ID` boundary, so distinct curves may retain identical observable rows.
- Equal rows per curve and exactly two curves per `Sweep_ID` give every curve and paired sweep equal nominal row weight.
- The tensor feature schema is exactly the approved observable allowlist. Identifiers, baseline, amplitude, label, and split remain in a row-aligned sidecar.

Splitting and perturbation:

- Both members of a `Sweep_ID` belong to one split, and held-out amplitudes form paired contiguous blocks.
- Clean, noisy, MR, and MRL derivatives reuse the same latent row identities and shared split manifest.
- Noise is applied after splitting with distinct train, validation, and test seeds.
- Runtime leakage checks reject duplicated row identities, group crossings, sidecar/tensor misalignment, wrong mass-node counts, invalid feature schemas, and clean/noisy mapping differences.

Model selection and evaluation access:

- Inner HPO folds are grouped and drawn only from the unscaled training artifact; preprocessing is fitted separately on each inner-fit fold, and every fit/score partition must contain both classes.
- Thresholds are selected from validation predictions, and metric definitions are explicit and consistent.
- Advanced test diagnostics are excluded by default and require explicit authorization.
- Source, configuration, dataset, split-manifest, feature-order, and scope metadata are written to run artifacts.

### Open acceptance gates

The following must still be satisfied before treating results as inferential evidence rather than a descriptive model-pair study:

- Report curve- and `Sweep_ID`-level metrics and confidence intervals clustered on `Sweep_ID`; state the effective group count explicitly.
- Disable high-capacity HPO for the 15-group run or label it exploratory, and predeclare the final model and single locked-test evaluation.
- Preserve and verify HPO/model lineage manifests whenever artifacts leave the managed run directory.
- Add fitted calibration on data independent of model selection before interpreting outputs as probabilities; specify a target class prior.
- Evaluate sensitivity across predeclared noise and covariance models, then validate on representative observational data.
- Require a locked test containing unseen hadronic baselines and unseen quark parameter tuples for any phase-general claim.

All current results and application text must remain labeled **APR-1-versus-fixed-CFL4 model-pair discrimination**. No output should be presented as an astrophysical `P(quark | observation)` under the current design.

## Regression-test and invariant status

Known explicit regression tests cover:

1. paired `Sweep_ID` membership and contiguous, disjoint amplitude blocks;
2. common-mass resampling with the configured row count; and
3. retention of identical observable rows belonging to distinct curves, demonstrating that global deduplication is absent.

Runtime dataset audits additionally enforce feature allowlisting, row-sidecar alignment, unique `Row_ID` values, curve/sweep/group split disjointness, paired amplitudes, common-grid counts, train-only scaling, and exact clean/noisy group and row alignment. These fail-closed artifact checks are valuable, but they do not replace focused unit tests for every invariant.

Regression coverage still required or requiring confirmation includes:

1. stale or mismatched HPO best-parameter artifacts are rejected after parameter, tensor, scaler, sidecar, transform, component, or feature-subset changes;
2. HPO fails if a future code path attempts to read validation or test artifacts as inner folds or fits preprocessing outside the inner-fit partition;
3. metric tests distinguish trapezoidal PR-AUC, average precision where used, quark-class F1, and macro-F1;
4. clustered intervals, once implemented, resample `Sweep_ID` rather than rows and are deterministic for a fixed seed;
5. every advanced test diagnostic path fails without authorization and records authorized access;
6. generation remains reproducible from its manifest, curve identifiers remain stable and unique, and rejected amplitudes cannot produce an unpaired dataset.

## Minimum defensible interpretation

If all engineering acceptance criteria pass but the experiment remains one baseline per class, the defensible conclusion is limited to:

> Under the declared APR-1 and fixed-CFL4 constructions, shared deformation family, amplitude support, mass grid, and synthetic observation model, the selected observables exhibit the reported degree of model-pair separability.

That statement does not imply generalization to other hadronic EoS, other quark parameterizations, hybrid stars, or real observational populations.
