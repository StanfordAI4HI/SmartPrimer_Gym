"""
This file contains code on how we generated the 10 random splits
"""

import random
import pandas as pd

df = pd.read_csv("./primer_data/new_data.csv")

# these two lines are no longer necessary
df['postive_feedback'] = df['postive_feedback'].apply(lambda x: 0 if x == 'None' else 1)
df['stage'].fillna(5, inplace=True)

student_ids = df['user_id'].unique().tolist()

for i in range(1, 11):
	random.shuffle(student_ids)
	train_ids = student_ids[:len(student_ids) // 2 + 1]
	valid_ids = student_ids[len(student_ids) // 2 + 1:]

	df[df['user_id'].isin(train_ids)].to_csv(f"./primer_data/train_df_new_new_split{i}.csv", index=False)
	df[df['user_id'].isin(valid_ids)].to_csv(f"./primer_data/valid_df_new_new_split{i}.csv", index=False)