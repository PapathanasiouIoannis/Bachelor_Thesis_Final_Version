# Family-held-out development model selection

No locked-test row or metric was used. Hyperparameters were tuned with
exhaustive out-of-family predictions on the 13 training groups, followed
by a single comparison of four finalists on the two validation groups.

| Finalist | Inner family accuracy | Validation family accuracy | Validation Brier | Max probability range across A |
|---|---:|---:|---:|---:|
| forest_mr_d2_l6 | 0.901 | 1.000 | 0.0000 | 0.0028 |
| forest_mrl_d2_l6 | 0.915 | 1.000 | 0.0000 | 0.0046 |
| logistic_mr_c0p1 | 1.000 | 1.000 | 0.0013 | 0.0331 |
| logistic_mrl_c0p001 | 1.000 | 1.000 | 0.1499 | 0.0245 |

Selected candidate: `logistic_mr_c0p1`.

Tune within training by exhaustive family-pair OOF CV; retain the best hyperparameters within 0.02 of the best family accuracy using the strongest regularization/shallowest forest; admit finalists within one validation curve of the best family-balanced accuracy; among candidates within 0.02 inner-CV accuracy, choose the lower-complexity model.

The selected specification must be committed as an immutable model profile
before the final test pair is opened exactly once.
