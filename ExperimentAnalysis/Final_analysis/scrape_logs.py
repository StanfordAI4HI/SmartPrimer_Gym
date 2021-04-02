import numpy as np
import pandas as pd



def read_from_txt():
	rows = np.arange(0,1500)
	cols = ['user_id', 'input_message_kid', 'stage', 'total_attempts_in_stage', 'grade_norm', 'pre-score_norm', 'stage_norm', 'failed_attempts_norm',
	        'pos_norm','neg_norm', 'hel_norm', 'anxiety_norm', 'action', 'output', 'postive_feedback', 'reward', 'grade', 'completed',
	        'pre', 'post', 'adjusted_score', 'anxiety', 'time_stored']

	data = pd.DataFrame(index=rows, columns=cols)

	text_file = open('logs.txt')
	lines = text_file.readlines()

	row = -1
	line = -1
	first = True

	while line < len(lines):
		if 'Update Starts' in lines[line]:
			line += 1
			kid_rows = []

			while 'Update' not in lines[line]:
				if 'Grade:' in lines[line]:
					grade = int(lines[line][7:].replace('\n',''))
					line+=1
				elif 'Completed' in lines[line]:
					completed = lines[line][11:].replace('\n','')
					line+=1
				elif 'Pre Score:' in lines[line]:
					pre_score = int(lines[line][11:].replace('\n',''))
					line+=1
				elif 'Post Score:' in lines[line]:
					post_score = int(lines[line][12:].replace('\n',''))
					line+=1
				elif 'Adjusted Score:' in lines[line]:
					adjusted_score = int(lines[line][16:].replace('\n',''))
					line+=1
				elif 'Anxiety:' in lines[line]:
					if 'None' in lines[line]:
						anxiety = None
					else:
						anxiety = int(lines[line][9:].replace('\n',''))
					line+=1
				else:
					line+=1

		elif 'Update' in lines[line] and '--------' in lines[line]:
			row += 1
			line += 1
			kid_rows.append(row)

			while '-----------' not in lines[line]:
				print(lines[line])
				if 'User: ' in lines[line]:
					data.iloc[row, 0] =  int([int(s) for s in lines[line].split() if s.isdigit()][0])
					if data.iloc[row, 0] == 97:
						a=1
					line += 1
				elif 'Input:' in lines[line] and not 'Tokenized' in lines[line] and not 'Next' in lines[line]:
					data.iloc[row, 1] = lines[line][7:].replace('\n','')
					line += 1
				elif 'Current Stage:' in lines[line]:
					data.iloc[row, 2] = [int(s) for s in lines[line].split() if s.isdigit()][0]
					line += 1
				elif lines[line][0:15] == 'Total Attempts:':
					data.iloc[row, 3] = [int(s) for s in lines[line].split() if s.isdigit()][0]
					line+=1
				elif 'Observation: ' in lines[line] and not 'Next' in lines[line]:
					if ']' in lines[line]:
						a = lines[line][14:-2].split()
						data.iloc[row, 4:12] = a
						line += 1
					else:
						a = lines[line][14:-1].split() + lines[line+1][:-2].split()
						data.iloc[row, 4:12] = a
						line+=2
				elif 'Action:' in lines[line]:
					if 'question' in lines[line]:
						data.iloc[row, 12] = 3
					elif 'hint' in lines[line]:
						data.iloc[row, 12] = 0
					elif 'nothing' in lines[line]:
						data.iloc[row,12] = 1
					elif 'encourage' in lines[line]:
						data.iloc[row,12] = 2
					else:
						data.iloc[row, 12] = [int(s) for s in lines[line].split() if s.isdigit()][0]

					line+=1
				elif 'Output:' in lines[line]:
					data.iloc[row, 13] = lines[line][8:].replace('\n','')
					line+=1
				elif 'Positive Feedback:' in lines[line]:
					data.iloc[row, 14] = lines[line][19:].replace('\n','')
					line += 1
				elif 'Reward:' in lines[line]:
					data.iloc[row, 15] = lines[line][8:].replace('\n','')
					line += 1
				else:
					line += 1

			data.iloc[row, 16] = grade
			data.iloc[row, 17] = completed
			data.iloc[row, 18] = pre_score
			data.iloc[row, 19] = post_score
			data.iloc[row, 20] = adjusted_score
			data.iloc[row, 21] = anxiety

		elif('---------- Agent is Stored' in lines[line]):
			line+=1


			if len(lines[line])!=0:
				stored_date = lines[line][14:]
			else:
				stored_date = 'none'

			# if not first:
			data.iloc[kid_rows, 22] = stored_date

			# else:
			# 	first = False

			# prev_stored_date = stored_date

		else:
			line+=1

	return data

def filter_on_indices(data, file_name):
	data = data.iloc[0:1116, :]
	realChildren = pd.read_csv(file_name).iloc[0:338, :]
	realChildren['UserID'] = realChildren['UserID'].astype(int)
	data = data[data['user_id'].isin(realChildren['UserID'])]

	print(data.shape)
	return data

def read_time_from_txt():
	id_time_mapping = []
	text_file = open('logs.txt')
	lines = text_file.readlines()

	for line in lines:
		if '{"to": 1, "from": 'in line:
			a = [x.strip(',') for x in line[10:25].split(' ')]
			b = [int(s) for s in a if s.isdigit()]
			id_time_mapping.append(b)
		if '"updated_at": "' in line:
			idx = line.find('"updated_at": "') + 15
			end_idx = line[(idx):].find('"') + idx
			id_time_mapping[-1].append(line[idx:end_idx])

	id_time_mapping = pd.DataFrame(id_time_mapping, columns=['id', 'updated_at'])
	return id_time_mapping

file_name = '/Users/williamsteenbergen/PycharmProjects/SmartPrimerFall/ExperimentAnalysis/Final_analysis/rlbot_results.csv'

id_time_map = read_time_from_txt()
data = read_from_txt()
data = filter_on_indices(data, file_name)

data = data[60:]
data.to_csv('offline_rl_data_with_times.csv')
a=1