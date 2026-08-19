# Streamlit deployment branch

This branch is an isolated runtime surface for the two legacy Streamlit entry
points:

- `app_ref.py`
- `perturb_app_ref.py`

It follows the current `main` source tree but intentionally restores only the
compact tensors, fitted scalers, model parameters, and model weights loaded by
those applications. The restored artifacts come from commit `d41f02e`, the
last pre-cleanup revision that powered the applications.

These files are deployment inputs for the historical interfaces. They are not
new scientific results, must not be merged back into `main`, and do not change
the repository's locked family evidence or artifact policy.
