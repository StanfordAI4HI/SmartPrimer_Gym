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

## Deployment

```python
from deploy import load_pytorch_agent, DeployPolicy

policy = load_pytorch_agent()
student1 = [
 [-1., -0.5,-0.33333333,-0.7,-1.,-1,1,-0.83333333],
 [-1,-0.5,0,-0.8, -1,-1,1,-0.833333],
 [-1,-0.5,0,-0.8, -1,0.076,-1,-0.833333],
 [-1,-0.5,0,-0.4, -1,-1,-1,-0.833333],
 [-1,-0.5,0,-0.4, -0.032, -0.29,-1,-0.8333333],
 [-1,-0.5,0, 0.2, -1, 0.334,-1,-0.833333],
 [-1,-0.5,0,0.2,1,-1,1,-0.8333333],
 [-1,-0.5,0,0.3,1,-1,1,-0.8333333]
]
print(["p_hint", "p_nothing", "p_encourage", "p_question"])
for obs in student2:
    print(policy.get_action(np.array(obs), temperature=0.9))
```