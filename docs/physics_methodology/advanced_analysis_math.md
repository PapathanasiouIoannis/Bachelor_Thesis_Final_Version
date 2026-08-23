# Advanced Diagnostics and Interpretability

The exploratory workflows include diagnostics in `src/ml/advanced/` for
inspecting fitted-model behaviour. These tools can reveal structure inside the
retained synthetic dataset or explain a classifier's response, but they do not
prove that the learned boundary represents a universal physical distinction.

- **UMAP projections:** Uniform Manifold Approximation and Projection provides a
  two-dimensional view of the retained feature space. Apparent separation is a
  descriptive property of that embedding and dataset; it is not evidence of
  phase-general discrimination or observational validity.
- **Brier scores and reliability diagrams:** These are calibration diagnostics
  for the evaluated synthetic split. They do not fit a calibration mapping and
  do not turn a classifier score into a real-world frequency or astrophysical
  posterior probability.
- **SHAP feature attribution:** Shapley Additive exPlanations describe how the
  fitted model's score changes with its inputs. They identify predictive
  dependence inside the retained experiment, not exact causal or physical
  contributions.

The controlling interpretation and remaining validation risks are documented
in [`../CLASSIFICATION_RISK_AUDIT.md`](../CLASSIFICATION_RISK_AUDIT.md).
