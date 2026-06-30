# Machine Learning Architecture (`src/ml/` and `src/ml_perturb/`)

The framework abandons simple linear models in favor of advanced, non-linear classification architectures. The primary orchestration leverages **XGBoost** and **PyTorch Multi-Layer Perceptrons (MLPs)**.

### The Objective
The core algorithmic objective is binary classification: determining the microscopic phase state (Hadronic vs. Quark) strictly from macroscopic observables $X = \{M, R, \log_{10}\Lambda\}$.

### The Observational Noise Framework
The true capability of this thesis lies in the perturbed inference pipeline (`perturb_main.py`). The framework explicitly confronts the "Masquerade Problem" by simulating real-world observational uncertainties. Gaussian noise is injected into the theoretical data arrays to replicate the error margins of modern observatories like NICER and LIGO.

The models are then aggressively retrained on this smudged dataset. The subsequent performance analysis fundamentally demonstrates that while $(M, R)$ alone suffer massive degradation under observational noise, the inclusion of Gravitational Wave priors ($\Lambda$) successfully disentangles the overlapping phase spaces.
