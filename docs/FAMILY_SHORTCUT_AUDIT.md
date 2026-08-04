# Family classification shortcut audit

This audit uses only the locked training and validation families. The test
pair remains unopened. Red probes are deliberately forbidden inputs; high
scores for direct label proxies demonstrate why metadata is isolated from
the observable feature tensors.

| Probe | Status | Balanced accuracy | ROC AUC |
|---|---|---:|---:|
| deformation_A_only | allowed_control_not_model_input | 0.500 | 0.500 |
| generation_controls | forbidden_metadata | 0.500 | 0.500 |
| serialization_geometry | forbidden_artifact | 1.000 | 1.000 |
| global_physics_summaries | forbidden_out_of_scope_physics | 1.000 | 1.000 |
| quark_parameter_presence | forbidden_direct_label_proxy | 1.000 | 1.000 |
| formula_provenance_flag | forbidden_direct_label_proxy | 1.000 | 1.000 |

## Structural checks

- `locked_test_not_used`: PASS
- `sample_identity_disjoint`: PASS
- `family_identity_disjoint`: PASS
- `amplitude_balanced`: PASS
- `forbidden_feature_overlap_absent`: PASS
- `no_exact_observable_duplicates`: PASS
- `A_only_at_chance`: PASS
- `positive_controls_detect_direct_proxies`: PASS

The deformation amplitude is balanced within every development split and
class, and its standalone classifier remains at chance. The production
models may consume only the explicitly listed radius and tidal features.
