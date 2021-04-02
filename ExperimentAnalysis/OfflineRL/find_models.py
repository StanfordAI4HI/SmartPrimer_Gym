import pandas as pd
import numpy as np

import os


def get_model_names(dir):
	file_names = os.listdir(dir)

	dates = []
	names = []
	for file_name in file_names:
		if '.meta' in file_name:
			names.append(file_name)
			dates.append(file_name[6:25])


	model_names = pd.DataFrame(list(zip(names, dates)), columns=['file_name', 'date'])
	model_names['date'] = pd.to_datetime(model_names['date'], utc=True)
	model_names = model_names.sort_values(['date'])
	model_names = model_names.reset_index()
	return model_names

def clean_data(timestamp_data, model_names):
	step2stage = {"height": 0, "width": 1, "length": 2, "volume": 3, "weight": 4, "comparison": 5, "finish": 6}

	timestamp_data['time_stamp'] = pd.to_datetime(timestamp_data['time_stamp'])

	#sort by id to add rewards
	data = timestamp_data.sort_values(['user_id','time_stamp'])
	data = data.reset_index()

	#Add final reward
	last_user = -1
	H=0
	for index, row in data.iterrows():

		if row['user_id'] != last_user:
			#add extra reward to previous row if kid finished
			if data.iloc[index-1, 7] == True and index !=0:
				data.iloc[index-1, -1] += data.iloc[index-1, 5] - (1 + 0.3) * H * 0.01

			elif index != 0:
				data.iloc[index - 1, -1] = -8

			#reset num hints
			H = 0

		if row['action'] == 'hint':
			H += 1

		last_user = row['user_id']

	#normalize the observation space
	data['grade'] = (data['grade'] - 3)/1
	data['pre_score'] = (data['pre_score']-4)/4
	data['failed_attempts'] = (data['failed_attempts']-10)/10
	data['anxiety'] = (data['anxiety']-27) / 18
	data['pos'] = (data['pos'] - 0.5) / 0.5
	data['neg'] = (data['neg'] - 0.5) / 0.5
	data['help'] = (data['help'] - 0.5) / 0.5


	for index, row in data.iterrows():
		data.iloc[index, 13] = (step2stage[row['stage']] - 3) / 3

	#add column of model names
	data['model_name'] = ''
	for index, row in data.iterrows():
		less_than = model_names[model_names['date'] < row['time_stamp']]['file_name']
		if len(less_than) != 0:
			data.iloc[index, -1] = less_than.values[-1]

	#remove the observations for which we do not have a model
	data = data[data['model_name'] != '']

	#remove observations for which we are insecure
	data = data[data['time_stamp'] > model_names.iloc[13, 2]]

	return data

if __name__ == '__main__':
	timestamp_data = pd.read_csv("action_selection.csv")
	directory = '/Users/williamsteenbergen/Documents/Stanford/SmartPrimer/Final_Logs/store'

	model_names = get_model_names(directory)
	cleaned_data = clean_data(timestamp_data, model_names)

	cleaned_data.to_csv('cleaned_data.csv')
	a=1