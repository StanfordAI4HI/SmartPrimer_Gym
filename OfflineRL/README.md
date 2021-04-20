# Offline Policy Evaluation and Optimization

We use the data generated and cleaned by preprocessing code stored in `../ExperimentAnalysis/OfflineRL`.

We are doing a 10 randomly generated 50/50 split of dataset to produce our training and validation dataset.
Note that this is not the same as cross-validation (which requires non-overlapping validation sets). 
Our choice is primarily motivated by the effective sample size of our Offline Policy Evaluation (OPE) procedure.

The algorithms and estimators implemented are in `offline_pg.py`

Implemented estimators are:

|                                                                 | Minibatch | Function Names   |
|-----------------------------------------------------------------|-----------|------------------|
| Importance Sampling (IS)                                        | Yes       | `is_ope`         |
| Clipped Importance Sampling (Clipped IS)                        | Yes       | `clipped_is_ope` |
| Weighted Importance Sampling (WIS)                              | No        | `wis_ope`        |
| Consistently Weighted Per-Decision Importance Sampling (CWPDIS) | No        | `cwpdis_ope`     |

The training is carried out by two main functions:
- `bc_train_policy`: Behavior cloning style pre-training.
- `offpolicy_pg_training`: Supports policy gradient style direct optimization of all estimators.
- `minibatch_offpolicy_pg_training`: Supports mini-batch style policy gradient optimizations of two estimators.

Other available utility functions:
- Action masking through `masked_softmax` method (during policy gradient)
- KNN-style action masking (by loading in externally trained KNN weights)

Plotting code upload (TBD).