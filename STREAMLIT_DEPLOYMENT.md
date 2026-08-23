# Streamlit deployment branch

This branch is an isolated runtime surface for two lasting public post-thesis
demonstrations:

- [Baseline synthetic-classifier demonstration](https://eoslab-clean-inference.streamlit.app/)
  (`app_ref.py`)
- [Perturbation-sensitivity demonstration](https://eoslab-perturbed-inference.streamlit.app/)
  (`perturb_app_ref.py`)

Both applications were developed by Ioannis Papathanasiou under the supervision
of Charalampos Moustakidis and Theodoros Diakonidis. Codex assisted with
software development; scientific interpretation and responsibility remain with
Ioannis Papathanasiou.

## Interpretation boundary

The applications expose classifier scores from retained restricted synthetic
comparisons. They are not observational inference services. Their scores are
not calibrated astrophysical probabilities, composition posteriors,
observational measurements, confidence intervals for stellar composition, or
evidence about the interior of a real star. Input uncertainty is represented by
independent Gaussian sampling rather than real NICER or LIGO posterior samples.
SHAP output describes fitted-model behaviour only.

It follows the current `main` source tree but intentionally restores only the
compact tensors, fitted scalers, model parameters, and model weights loaded by
those applications. The restored artifacts come from commit `d41f02e`, the
last pre-cleanup revision that powered the applications.

These files are deployment inputs for the public demonstrations. They are not
new scientific results, must not be merged back into `main`, and do not change
the repository's locked family evidence or artifact policy.

The deployment branch must retain normal Streamlit web protections and must not
disable CORS or XSRF protection. A deployment is ready for promotion only when
the branch-specific artifact contract, scientific-language tests, and complete
test suite pass together.
