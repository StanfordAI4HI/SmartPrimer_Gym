import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("offline_with_probabilities.csv")
data = data[40:]


beh_prob = []

for index, row in data.iterrows():
	probs = row[['p_hint', 'p_nothing', 'p_encourage', 'p_question']]
	beh_prob.append(probs[row['action']])

data['beh_prob'] = np.array(beh_prob)

# data = data.groupby('user_id').mean()
data = data.groupby('user_id', as_index=False)['beh_prob'].mean()
plt.hist(data['beh_prob'], bins=30)
plt.title('histogram of mean probabilities for children')
plt.ylabel('frequency (# children)')
plt.xlabel('median probability of what happened per child')
plt.show()
a=1