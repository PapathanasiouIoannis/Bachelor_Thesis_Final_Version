# Safe family workflow architecture

The supported family-classification workflow enters through `eoslab.py` and
composes through `src/family_workflow.py`. The workflow is deliberately split
into a compatibility and safety facade plus focused internal leaves. This
document records the boundary so future maintenance does not hide the
one-shot evidence gates or subprocess ordering behind generic helpers.

## Responsibility map

| Module | Owned responsibility |
| --- | --- |
| `src/family_workflow.py` | Supported API, canonical constants/errors/path record, path resolution, live dependency wiring, final-evaluation refusal, and the visible development lifecycle |
| `src/family_runner/evidence.py` | Regular-file JSON reads, newline-portable hash matching, development-artifact summaries, and compact final-evidence integrity/status projection |
| `src/family_runner/status.py` | Profile loading and cross-identity/hash validation, family-split summaries, and read-only workflow-status assembly |
| `src/family_runner/development.py` | Pure construction of the fixed five-stage development command plan |
| `src/family_runner/__init__.py` | Inert package boundary; it performs no eager leaf imports |

The implementation graph is intentionally one-way:

```text
eoslab.py
    -> src.family_workflow
        -> framework.family_pilot
        -> src.ml.family_splitting
        -> src.ml.family_final
        -> family_runner.development
        -> family_runner.evidence
        -> family_runner.status
```

The three `family_runner` leaves currently use only the standard library. The
durable architecture rule is that they remain mutually independent and never
import the facade. They are maintained implementation boundaries, not
independent public entry points.

## Supported surface and facade policy

New supported work should enter through `eoslab.py`. Direct Python callers use
the constants, errors, `FamilyWorkflowPaths`, status operation, guards, and
development operation exported by `src.family_workflow`.

Operations remain real facade functions with their established signatures.
They delegate through imported leaf modules and pass the facade's current
loaders, hash matcher, profile-entry provider, reporting constants, and Python
executable at call time. This preserves the compatibility seams and leaf
reload behavior without creating a leaf-to-facade edge.

The facade intentionally retains its existing import timing. Default profile
and output paths are absolute objects captured when the module and function
defaults are created. The profile loaders also retain their established eager
imports and shared configuration timing. Changing `PROJECT_ROOT` or `CONFIG`
after import is not a supported way to relocate or reconfigure an active
workflow.

## Read-only status and evidence boundary

`family_workflow_status` never loads or scores final-test tensors and never
returns saved predictions. Its durable order is:

1. Reject simultaneous `paths` and individual path options.
2. Resolve paths when an explicit `FamilyWorkflowPaths` record was not passed.
3. Load generation, split, and locked model profiles in that order.
4. Validate generation/split/model identities, then validate the split-profile
   hash recorded by the locked model profile.
5. Build the split and generation summaries.
6. Read development artifacts, compact development evidence, and final-test
   evidence in that order.

Profile failure can therefore prevent status projection even when a final lock
exists. Changing that precedence is a behavior and diagnostics decision, not a
structural refactor.

## Development lifecycle ownership

`run_family_development` remains a visible state machine in the facade. Its
order is part of the safety contract:

1. Reject simultaneous path inputs.
2. Refuse any normalized stage name containing `final`, plus the exact aliases
   `test`, `evaluation`, and `score_test`.
3. Coerce and validate the jobs and permutation counts.
4. Resolve paths, load the three profiles, and validate their identities and
   the locked split hash.
5. Read final evidence and fail closed if the one-shot test was opened or any
   final evidence is incomplete, unreadable, or inconsistent.
6. Build the fixed command plan: generation, curve preparation, shortcut
   audit, model selection, then robustness. Only generation may receive
   `--force-regenerate`.
7. Immediately before each launch, require that stage's entry point; then run
   it synchronously with captured output. The first nonzero exit stops later
   stages, and completed earlier stages are not rolled back.
8. After all five stages succeed, perform a fresh full status read.

The facade never imports or launches `family_final_test.py`. The public final
request operation always refuses; the separately governed script remains the
only owner of the exclusive one-shot opening sequence.

## Locked-evidence invariants

The core recorded chain includes:

- `data/family_pilot_v1/family_ml/LOCKED_TEST_OPENED.json`;
- `docs/family_final_test.json`;
- `framework/family_model_profile.json` and its locked split-profile hash;
- `docs/family_shortcut_audit.json`;
- `docs/family_model_selection.json`; and
- `docs/family_development_robustness.json`.

The final script creates the marker exclusively before it reads the test
tensors. Marker creation is the opening event, so a crash after creation still
consumes the attempt. Generic cleanup must never delete, regenerate, reformat,
or relocate the marker or its hash-bound evidence.

Any marker/result read error or evidence presence makes rerun permission false.
A successfully parsed marker object reports `LOCKED_TEST_OPENED` even when its
integrity is invalid. Unreadable, malformed, or non-regular marker evidence
remains fail-closed without receiving that state label. A valid completed
record requires a `COMPLETED` marker, an archived result with
`test_open_count == 1`, a matching result hash, matching recorded commit and
model-profile identities, the actual locked model-profile hash, and safe,
present development evidence matching the model profile's hashes.

Hash portability accepts only the raw, LF, or CRLF byte representation of a
JSON file. It does not canonicalize JSON values, whitespace, or key order. The
current checked-in state must continue to report `LOCKED_TEST_OPENED`, open
count one, valid integrity, no integrity errors, and no rerun permission.

## Structural invariants

Future structural work must preserve all of the following unless a separate
behavior-change proposal explicitly replaces them:

- `eoslab.py` continues to enter through the facade;
- `src.family_runner` remains inert and every leaf remains independently
  importable;
- no family leaf imports `src.family_workflow` or another leaf;
- facade wrapper functions retain their documented signatures and use current
  facade dependencies per call;
- canonical constants, errors, and `FamilyWorkflowPaths` remain facade-owned;
- status remains compact and never reads final tensors or exposes predictions;
- the development plan never includes the final-test script;
- the final-evidence gate remains before command planning and subprocess work;
- development remains sequential and fail-fast with a fresh status read only
  after five successful stages; and
- structural changes never rewrite locked profiles, evidence, or the
  classification policy configuration.

## Deferred behavior work

The structural split does not declare every inherited behavior ideal. These
items require separate behavior proposals and dedicated tests:

- the final marker or profiles can change after the parent's pre-run checks and
  before or during child execution;
- a stage exit code of zero is accepted without a dedicated completeness check
  for every expected artifact, and partial successful stages are not rolled
  back;
- jobs, permutations, and force values retain their characterized legacy
  `int()` / `bool()` coercion behavior;
- changing only `project_root` does not relocate the absolute default profile,
  data, and report paths captured at import;
- a post-run status failure occurs after all five child stages have completed;
- a direct Python call does not perform the full family TOML policy validation
  owned by the supported launcher path;
- status validates profiles before it can report an already existing final
  lock;
- workflow status permits raw/LF/CRLF hash equivalence, while older post-test
  consumers use raw file hashes; and
- crash/concurrency semantics of the exclusive final-opening script remain a
  separately governed test and policy boundary.

Treat these as behavior changes, not reasons to move more safety sequencing out
of the facade.
