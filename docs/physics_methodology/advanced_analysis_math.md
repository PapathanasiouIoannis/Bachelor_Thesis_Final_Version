# Advanced Diagnostics and Interpretability (`src/ml/advanced/`)

To validate the algorithmic robustness of the machine learning classifiers, the framework executes a suite of stringent mathematical diagnostics:

- **UMAP Topology Projections:** Uniform Manifold Approximation and Projection (UMAP) mathematically reduces the highly dimensional $M-R-\Lambda$ space into 2D, empirically proving that XGBoost has successfully drawn distinct, non-linear class boundaries.
- **Brier Score Calibration:** Utilizing reliability diagrams to assess whether the model's output probability strictly adheres to frequentist likelihood (e.g., evaluating if $P(\text{Quark}) = 0.8$ actually correlates to an 80% observational frequency).
- **SHAP Feature Importance:** Utilizing Shapley Additive exPlanations (Lundberg & Lee, 2017) derived from cooperative game theory to formally quantify the marginal predictive contribution of each feature, proving that $\Lambda$ is the critical determinant in the phase decision boundary.
