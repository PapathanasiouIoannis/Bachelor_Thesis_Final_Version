# Runtime artifacts and locked evidence

The repository distinguishes generated runtime files from intentionally
versioned scientific evidence. This boundary keeps routine physics and machine
learning runs from modifying source-control state while preserving the one-shot
family-test record.

## Ignored runtime roots

The following roots contain reproducible or machine-local outputs and are not
versioned:

- `data/`, except for the final-test lock marker described below;
- `models/`;
- `outputs/` and `outputs_perturb/`;
- `plots/` and `plots_perturb/`; and
- `runs/`.

These paths may contain Parquet tables, tensor splits, scalers, optimization
results, model weights, probabilities, metrics, plots, logs, and worker chunks.
Their bytes can vary across dependency versions and hardware even when their
scientific content is equivalent. They are operational artifacts, not source.

Tests must write generated files beneath pytest temporary directories. A new
small deterministic test input belongs in `tests/fixtures/`, not in a runtime
root.

## Protected final-test marker

The sole versioned exception inside `data/` is:

```text
data/family_pilot_v1/family_ml/LOCKED_TEST_OPENED.json
```

This file is the exclusive one-shot opening marker for the recorded family
test. It binds the archived final result and locked model profile by SHA-256.
The model profile in turn binds the shortcut audit, model-selection record, and
development-robustness record. Changing whitespace, key order, content, path,
or filename can invalidate that evidence chain.

Generic cleanup must never recursively remove or regenerate the marker. The
family status command must continue to report `LOCKED_TEST_OPENED`, an open
count of one, valid integrity, and no rerun permission.

## Checked-in evidence under `docs/`

Compact CSV, JSON, NPZ, PNG, and Markdown files under `docs/` may be versioned
when they are deliberately promoted evidence for a documented audit. Promotion
requires:

1. the exact configuration or profile identity;
2. provenance and interpretation boundaries;
3. integrity hashes when the evidence participates in a locked workflow; and
4. an authored document explaining what the files establish and do not
   establish.

Audit output should otherwise be generated in an ignored, non-overwriting run
directory. A future experiment must use a separately versioned evidence record
instead of overwriting the frozen family-v1 paths.

## Rebuilding historical runtime artifacts

The historical launchers are compatibility workflows and require the complete
legacy dependency stack:

```powershell
py -3 -m pip install -r requirements.txt
```

The normal dependency order is:

1. Generate the legacy physics dataset with `physics_main.py`.
2. Build clean ML artifacts with `main.py`.
3. Build perturbed ML artifacts with `perturb_main.py`.
4. Start `app_ref.py` or `perturb_app_ref.py` only after their required model,
   scaler, and dataset paths exist.

Readiness-sized runs can use the launchers' `--smoke-test` options. A complete
historical regeneration can be expensive and may not be byte-identical to an
older environment. Each launcher records its active data root, and missing
prerequisites fail with an explicit path list.

The supported post-thesis `eoslab.py` workflow does not depend on pre-populated
legacy model artifacts. Its managed run directories are unique and ignored,
and compact terminal summaries are promoted explicitly with `export-summary`.

## Source-control guard

The regression suite inspects `git ls-files` and permits exactly one tracked
path beneath the runtime roots: the final-test marker. It also verifies that the
marker is explicitly unignored and that the checked-in evidence chain remains
valid. Adding a new runtime artifact to Git therefore requires an intentional
policy change rather than an accidental force-add.
