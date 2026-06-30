# Machine Learning Architecture and the Masquerade Problem

The classification phase is designed to differentiate hadronic neutron stars from exotic quark stars. Because the underlying thermodynamics are hidden within the stellar core, the algorithm must deduce the internal phase state purely from macroscopic observables: $X = \{M, R, \log_{10}\Lambda\}$.

The orchestrator utilizes advanced non-linear frameworks, specifically extreme gradient boosting (XGBoost) and PyTorch-based Bayesian Neural Networks (BNNs).

### Confronting the Masquerade Problem
The core thesis revolves around the "Masquerade Problem" first popularized by Alford and others. Theoretical models demonstrate that hybrid stars and pure hadronic stars can exhibit mass-radius ($M-R$) relationships that are practically identical. This topological degeneracy makes it theoretically impossible to identify a quark star using mass and radius alone.

The actual test of this thesis occurs in `src/ml_perturb/perturb_main.py`. Real astrophysics is heavily constrained by observational uncertainties. Telescopes like NICER can only measure stellar radii with significant error bars. To replicate this, the script injects aggressive Gaussian noise directly into the theoretical $M$ and $R$ columns. 

By retraining the classifiers on this blurred, chaotic dataset, the analysis definitively proves that the mass-radius degeneracy becomes insurmountable. Only by providing the machine learning models with the tidal deformability parameter $\Lambda$ (as measured by gravitational wave interferometers like LIGO) can the algorithm successfully map and break the phase degeneracy.
