# Machine Learning Architecture

The classification phase of the thesis aims to differentiate hadronic from quark matter based strictly on macroscopic observables. The models trained are XGBoost and PyTorch Multi-Layer Perceptrons.

The input vector is defined as $X = \{M, R, \log_{10}\Lambda\}$.

The most important part of this analysis is the perturbed pipeline. Telescopes do not measure mass and radius perfectly. To address the masquerade problem, `perturb_main.py` takes the theoretical dataset and injects realistic Gaussian noise into the mass and radius measurements, similar to the error bars seen from NICER. By retraining the classifiers on this smeared data, the results show that tidal deformability $\Lambda$ is practically required to maintain any meaningful classification accuracy, since the $M-R$ space becomes hopelessly overlapping.
