import pandas as pd
import numpy as np

offline_data_file = '/Users/williamsteenbergen/PycharmProjects/SmartPrimerFall/gym_SmartPrimer/examples/offline_with_probabilities.csv'
data_with_probs = pd.read_csv(offline_data_file)
print('unique id: {}'.format(data_with_probs['user_id'].nunique()))


offline_data_file = '/Users/williamsteenbergen/PycharmProjects/SmartPrimerFall/ExperimentAnalysis/Final_analysis/offline_rl_data.csv'
data_without_probs = pd.read_csv(offline_data_file)
print('unique id: {}'.format(data_without_probs['user_id'].nunique()))

offline_data_file = '/Users/williamsteenbergen/PycharmProjects/SmartPrimerFall/ExperimentAnalysis/Final_analysis/offline_rl_data_with_times.csv'
data_without_probs = pd.read_csv(offline_data_file)
print('unique id: {}'.format(data_without_probs['user_id'].nunique()))

prev_id = -1

policies = []
policy = np.array([-1,-1,-1,-1, -1])
improvement = 0
for idx, row in data_with_probs.iterrows():
	if row['user_id'] != prev_id:
		policies.append(list(policy[0:4]/sum(policy[0:4])) + [policy[4], improvement])
		policy = np.array([0, 0, 0, 0, row['user_id']])

		policy[row['action']] += 1

		prev_id = row['user_id']

	else:
		policy[row['action']] += 1
		improvement = row['adjusted_score']

policies = pd.DataFrame(policies, columns=['hint', 'nothing', 'encourage', 'question', 'user_id', 'adjusted_score'])

only_hints = policies[policies['hint']==1]
only_question = policies[policies['question']==1]
only_encourage = policies[policies['encourage']==1]
only_nothing = policies[policies['nothing']==1]

not_hints = policies[policies['hint']!=1]
random = policies[(policies['hint'] < 0.4) & (policies['nothing'] < 0.4) & (policies['encourage'] < 0.4) & (policies['question'] < 0.4)]



print(np.mean(only_hints['adjusted_score']))
print(np.mean(policies['adjusted_score']))
print(np.mean(not_hints['adjusted_score']))
print(np.mean(only_question['adjusted_score'])) #1.58 but only 12 obs
print(np.mean(only_nothing['adjusted_score'])) #0.333 but only 3 obs
print(np.mean(only_encourage['adjusted_score'])) #2.25 but only 8 obs



a=1
