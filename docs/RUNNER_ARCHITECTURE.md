# Controlled runner architecture

The supported controlled pair workflow enters through `eoslab.py` and composes
through `src/physics/experiment_runner.py`. The runner is deliberately split
into a compatibility and lifecycle facade plus focused internal leaves. This
document records that boundary so later maintenance does not recreate a
monolith or hide durable run-state transitions behind generic helpers.

## Responsibility map

| Module | Owned responsibility |
| --- | --- |
| `src/physics/experiment_runner.py` | Supported API, compatibility identities and signatures, live dependency wiring, process orchestration, clocks, top-level persistence/effect ordering, and the running/terminal/failed lifecycle |
| `src/physics/runner/settings.py` | Numerical preset snapshots and resolved numerical/quark settings |
| `src/physics/runner/preflight.py` | Pair validation, common amplitude support, baseline recovery, provenance, and the `PairPreflight` record |
| `src/physics/runner/generation.py` | One worker's paired EoS/stellar generation protocol and stage-specific rejection records |
| `src/physics/runner/convergence.py` | Numerical refinement checks, physical screens, and convergence records |
| `src/physics/runner/artifacts.py` | Resolved TOML rendering, frame concatenation, and artifact-hash inventory |
| `src/physics/runner/run_logs.py` | Process-isolated worker log paths and deterministic log merging |
| `src/physics/runner/manifest.py` | Pure manifest construction and terminal-status selection |

The implementation graph is intentionally one-way:

```text
eoslab.py
    -> src.physics.experiment_runner
        -> runner.artifacts
        -> runner.convergence
        -> runner.generation
        -> runner.manifest
        -> runner.preflight -> runner.settings
        -> runner.run_logs
        -> runner.settings
```

No module under `src/physics/runner/` may import the facade. The package
`src.physics.runner` performs no eager imports. These two rules keep leaf-first
imports acyclic and allow the facade to inject current compatibility bindings
without a back-edge.

## Supported surface and compatibility policy

New controlled work should import `run_pair_experiment` or
`validate_pair_experiment` from `src.physics.experiment_runner`, normally via
`eoslab.py`. The leaf modules are maintained implementation boundaries, not
independent public entry points.

The facade uses two compatibility patterns:

- Canonical immutable records, errors, and schema constants are identity
  aliases, including `PairPreflight`, `PairGenerationError`, and
  `CONVERGENCE_COLUMNS`.
- Operations remain real facade functions with their established signatures.
  Where needed, they build dependency contexts at call time and then delegate
  through the imported leaf module. This preserves live configuration, test
  seams, reload behavior, and existing private imports while keeping
  implementations out of the facade.

`NUMERICAL_PRESETS` is intentionally a facade import-time snapshot of
`CONFIG`. Reloading the facade rebuilds that snapshot and synchronizes the
settings/preflight leaves. Changing `CONFIG` in place during an active run is
not a supported way to alter a resolved experiment.

## Parent/worker boundary

Joblib submits the top-level facade function `_generate_pair`, not a closure,
leaf alias, or prebuilt dependency bundle. Each child process constructs the
generation dependency context inside that facade call. Consequently:

- worker log paths use the child process identifier;
- the delayed payload contains only serializable run values;
- the facade remains the stable Windows/loky resolution point; and
- dependencies are resolved from the child process's current facade bindings.

Do not move dependency-context creation into the parent or submit a lambda,
partial, dataclass bundle, or direct leaf function to `delayed`.

## Lifecycle ownership

`run_pair_experiment` remains a visible state machine in the facade. Its order
is part of the run-record contract:

1. Validate and resolve the configuration, apply a valid recorded runtime
   override, create the run layout, configure logging, and write the resolved
   TOML.
2. Capture provenance and write the `running` manifest checkpoint. Both startup
   log messages still occur before the recovery `try` block.
3. Run workers, merge worker logs, assemble tables, evaluate convergence,
   persist all tables, select the status, create plots and the report, and hash
   artifacts.
4. Write the terminal manifest before returning a completed run or raising for
   `completed_with_rejections` / `failed_convergence`.
5. Recover only ordinary `Exception` failures. Recovery merges worker logs,
   rereads the on-disk manifest, and changes it to `failed` only while its
   status is still `running`; when recovery succeeds, the original exception is
   then re-raised.

`BaseException` subclasses bypass recovery. A deliberate exception raised
after a non-success terminal manifest does not overwrite that terminal status
with `failed`. The normal and recovery worker-log merge sites, the on-disk
status guard, clock/provider order, and bare re-raise are intentionally kept in
the facade.

## Structural invariants

Future structural changes must preserve all of the following unless a separate
behavior-change proposal explicitly replaces them:

- `eoslab.py` continues to enter through the facade;
- runner leaves never import `src.physics.experiment_runner`;
- the facade remains the joblib worker target and constructs worker contexts in
  the child;
- the `running` checkpoint stays before the recovery boundary;
- artifact generation and hash-inventory construction occur before the terminal
  checkpoint;
- recovery consults the persisted manifest and updates only `running` status;
- terminal rejection/convergence states are durable before their exception;
- the facade's exported names, canonical identities, and documented function
  signatures remain stable; and
- numerical settings, scientific schemas, and claim boundaries do not change
  in a structural PR.

## Deferred reliability work

The structural split does not declare every existing lifecycle behavior ideal.
The following are separate reliability changes because each alters observable
failure handling and needs its own tests and review:

- an unexpected worker error can bypass worker-log closure, and the main run
  does not own an outer log-handler `finally`;
- a worker log retained after an unlink `PermissionError` can be appended again
  by the recovery merge;
- a recovery merge/read/write failure can replace the primary exception and
  leave the persisted status at `running`;
- failures before the recovery boundary can leave no manifest or a durable
  `running` checkpoint; and
- artifact hashing currently omits missing fixed outputs instead of failing
  closed, while only plots are intentionally optional.

Treat these as behavior fixes, not opportunities to move more lifecycle code
out of the facade.
