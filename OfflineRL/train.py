from offline_pg import MLPPolicy, bc_train_policy, ope_evaluate, wis_ope

import pandas as pd

set_random_seed(72)

new_df = pd.read_csv("children_probs.csv")

old_names = ['grade_norm', 'pre-score_norm', 'stage_norm', 'failed_attempts_norm',
             'pos_norm', 'neg_norm', 'hel_norm', 'anxiety_norm']

new_names = ['grade', 'pre_score', 'stage', 'failed_attempts', 'pos', 'neg', 'help', 'anxiety']

for new_name, old_name in zip(new_names, old_names):
	new_df[old_name] = new_df[new_name]

new_df['adjusted_score'] = new_df['adjusted_improvement']

new_df['pre'] = new_df['pre-score_norm'] * 4 + 4
new_df['anxiety'] = new_df['anxiety_norm'] * 18 + 27

action_names = ['hint', 'nothing', 'encourage', 'question']
new_df['action_names'] = new_df['action']
new_df['action'] = new_df['action_names'].map(action_names.index)

bc_policy = MLPPolicy([10, 4, 4])
bc_train_losses, bc_train_esss, bc_val_losses, bc_val_esss, train_weights, valid_weights, train_opes, val_opes = bc_train_policy(
							bc_policy, new_df,
							new_df,
							epochs=1,  # 40,
							lr=1e-3,
							verbose=False, early_stop=False,
							train_ess_early_stop=25, val_ess_early_stop=15,
							return_weights=True,
							model_round_as_feature=False)

full_bc_full_ope, _, _ = ope_evaluate(bc_policy, new_df, wis_ope, 0.01, is_train=False,
                                      return_weights=True, use_knn=False)

print(full_bc_full_ope)