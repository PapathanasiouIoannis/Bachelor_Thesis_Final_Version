# Diagnostics and Interpretability

To prove the machine learning models aren't just memorizing data, the code runs several advanced mathematical diagnostics.

- **UMAP Projections**: This reduces the highly dimensional $M-R-\Lambda$ space down to 2D. It visually proves that the classifiers have found distinct nonlinear boundaries between the hadronic and quark classes.
- **Brier Score**: This measures calibration. A reliability diagram checks if a predicted probability of 90% actually corresponds to a 90% true positive rate, rather than just being an overconfident guess.
- **SHAP Values**: Based on cooperative game theory, Shapley values calculate the exact marginal contribution of each variable. The SHAP plots explicitly prove that $\log_{10}\Lambda$ is the dominant feature driving the model's decisions, especially when noise is introduced to the radius measurements.
