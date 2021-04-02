import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
import pandas as pd
from numpy import std, mean, sqrt

import scipy
sns.set_theme(style="ticks", color_codes=True)

def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (mean(x) - mean(y)) / sqrt(((nx-1)*std(x, ddof=1) ** 2 + (ny-1)*std(y, ddof=1) ** 2) / dof)


def processHint2Action(filename):
	with open(filename) as f:
		hint2actionRaw = json.load(f)

	hint2action = []
	for hintID in hint2actionRaw['data']['hint']:
		hint2action.append([hintID['id'], hintID['step_id']])
	hint2action = np.array(hint2action)
	return hint2action

def map2action(hintID, hint2action):
	actionID = hint2action[hint2action[:,0]==hintID, 1][0]
	if actionID < 10:
		action = 'hint'
	elif actionID == 10:
		action = 'temp'
	elif actionID == 13:
		action = 'nothing'
	elif actionID == 15:
		action = 'encourage'
	elif actionID == 16:
		action = 'question'
	return action

def getData(hint2action, realChildrenFileName, actionsTakenFileName, finishedChildrenFileName):
	# realChildren = np.loadtxt(realChildrenFileName)

	realChildren = pd.read_csv(realChildrenFileName).dropna()
	finishedChildrenTemp = realChildren[realChildren['Complete']=='Y']
	realChildren = np.array(realChildren['UserID']).astype(int)

	with open(finishedChildrenFileName) as f1:
		finishedChildren = json.load(f1)

	finishedChildrenList = []
	for temp in finishedChildren['data']['test']:
		if temp['adjusted_score'] != None:
			finishedChildrenList.append(temp['user_id'])

	with open(actionsTakenFileName) as f:
		actions = json.load(f)

	data = []

	for datapoint in actions['data']['response']:
		if datapoint['input_message']['from'] in realChildren and datapoint['input_message']['from'] in finishedChildrenList:
			action = map2action(datapoint['hint_id'], hint2action)
			if datapoint['positive_feedback'] != None:
				data.append([action, datapoint['positive_feedback'], datapoint['input_message']['from'], datapoint['created_at']])
			else:
				data.append([action, False, str(datapoint['input_message']['from']), datapoint['created_at']])

	data = np.array(data)
	data = pd.DataFrame(data, columns=['action', 'positive_feedback', 'user_id', 'time'])
	data = data.sort_values('time')

	print(data['user_id'].nunique())
	return data

def bootstrap(data, B, modelNumber):
	adjustedScores = []

	oldID = -1
	for i, id in enumerate(data['user_id']):
		if id != oldID:
			adjustedScores.append(data.iloc[i,:]['score'])
		oldID = id

	means = []
	for i in range(0, B):
		sampleIdx = np.random.randint(0, len(adjustedScores), size=len(adjustedScores))
		scores = np.array(adjustedScores, dtype=float)[sampleIdx]
		meanB = np.mean(scores)
		means.append(meanB)

	plt.hist(means)
	plt.title('Histogram for means with {} children'.format(modelNumber))
	plt.show()

	# print('For {} children we have an std of {}'.format(modelNumber, np.std(means)))
	return np.std(means)

def getAverageList(list, over):
	averageList = []

	for i in range(1, len(list)):
		if i <= over:
			print(list[0:i])
			averageList.append(np.mean(list[0:i]))
		else:
			averageList.append(np.mean(list[(i-over):i]))

	return averageList

def getReward(data):
	if data['action'] == 'hint':
		if data['positive_feedback']:
			return 0.11
		else:
			return 0.1
	elif data['positive_feedback']:
		return 0.1
	else:
		return 0


def getData2(childrenData, actionData, realChildrenFileName, finishedChildrenFileName):
	# realChildren = np.loadtxt(realChildrenFileName)
	realChildren = pd.read_csv(realChildrenFileName).dropna()
	realChildren = np.array(realChildren['UserID']).astype(int)

	with open(finishedChildrenFileName) as f1:
		finishedChildren = json.load(f1)

	finishedChildrenList = []
	for temp in finishedChildren['data']['test']:
		if temp['adjusted_score'] != None:
			finishedChildrenList.append(temp['user_id'])

	with open(childrenData) as f1:
		childrenData = json.load(f1)

	children = []
	for datapoint in childrenData['data']['user']:
		if len(datapoint['test']) != 0 and datapoint['id'] in realChildren and datapoint['id'] in finishedChildrenList:
			children.append([str(datapoint['id']), datapoint['test'][0]['pre_score'],
			                 float(datapoint['test'][0]['adjusted_score']), datapoint['grade'],
			                 datapoint['created_at'], ])


	children = np.array(children)
	children = pd.DataFrame(children, columns=['user_id', 'pre_score', 'adjusted_score', 'grade', 'created_at'])
	# data = pd.DataFrame(actionData, columns=['action', 'positive_feedback', 'user_id', 'time'])

	#join by child id

	final = pd.merge(children, actionData, how='left', on='user_id')
	final = final.sort_values(by='created_at', ignore_index=True)
	# final = final.iloc[85:, :]
	final = final.dropna()

	print('max is: {}'.format(np.max(final['adjusted_score'])))
	print('number of unique children is: {}'.format(len(final.user_id.unique())))

	perfectPost = []
	perfectPostTemp = 0
	old = -1
	counter = 0
	totalPerfect = 0
	user_id_counter = 0
	tempData = []

	grades = []
	modelNumber = 0
	seenOld = []
	adjusted_imp = []
	rewards = []
	psai = 0.3
	pre_scores = []

	nHints = 0

	childRewards = []
	childRewardPlotable = []
	for i, user_id in enumerate(final['user_id']):
		tempData.append(list(final.iloc[i, :]))
		childRewards.append(getReward(final.iloc[i, :]))

		if user_id not in seenOld:
			pre_scores.append(float(final.iloc[i,:]['pre_score']))

			grades.append(int(final.iloc[i,:]['grade']))
			adjusted_imp.append(float(final.iloc[i,:]['adjusted_score']))
			rewards.append(float(final.iloc[i,:]['adjusted_score']) - (nHints*0.01*psai) + getReward(final.iloc[i, :]))

			childRewards.append(rewards[-1])
			childRewardPlotable.append(np.sum(childRewards))
			childRewards = []

			nHints = 0
			seenOld.append(user_id)
			user_id_counter += 1
			counter+=1

			if int(final.iloc[i,:]['pre_score']) == 8:
				perfectPostTemp += 1
				totalPerfect += 1

			if counter == 5:
				modelNumber += 1
				perfectPost.append(perfectPostTemp)
				perfectPostTemp = 0
				counter = 0
				# tempData = pd.DataFrame(tempData, columns=final.columns)
				# print('ModelNumber {}: {}'.format(modelNumber, bootstrap(tempData, 10000, modelNumber)))

				tempData = []
		else:
			rewards.append(getReward(final.iloc[i, :]))

			if rewards[-1] == 0.01 or rewards[-1] == 0.11:
				nHints +=1

	# plt.plot(perfectPost)
	# plt.title('Number of children that had perfect pre score')
	# plt.xlabel('Model update')
	# plt.ylabel('Number of children that had perfect pre score')
	# plt.plot()

	# plt.plot(getAverageList(grades, 5))
	# plt.title('Grades averaged over last 5 children')
	# plt.xlabel('Child')
	# plt.ylabel('Grade average')
	# plt.plot()
	#
	# plt.plot(getAverageList(pre_scores, 5))
	# plt.title('pre score averaged over last 5 children')
	# plt.xlabel('Child')
	# plt.ylabel('pre score average')
	# plt.plot()

	nAverage = 10
	plt.plot(getAverageList(adjusted_imp, nAverage))
	plt.title('Adjusted score averaged over last {} children'.format(nAverage))
	plt.xlabel('Child')
	plt.ylabel('Adjusted score average')
	plt.plot()
	#

	# nAverage = 5
	# plt.plot(getAverageList(childRewardPlotable, nAverage))
	# plt.title('Rewards averaged over last {} children'.format(nAverage))
	# plt.xlabel('Child')
	# plt.ylabel('Child average')
	# plt.plot()
	#
	# nAverage = 50
	# plt.plot(getAverageList(rewards, nAverage))
	# plt.title('Rewards averaged over last {} actions'.format(nAverage))
	# plt.xlabel('Action')
	# plt.ylabel('Action average')
	# plt.plot()

	# plotAllActions(final['action'])
	# plotActionTimeline(final['action'], 30)

	print(perfectPost)
	a=1

def getDataControl(childrenData, actionData, realChildrenFileName, finishedChildrenFileName):
	realChildren = np.loadtxt(realChildrenFileName)

	with open(childrenData) as f1:
		childrenData = json.load(f1)

	children = []
	for datapoint in childrenData['data']['user']:
		if len(datapoint['test']) != 0 and datapoint['id'] in realChildren and datapoint['test'][0]['post_score'] != None:
			children.append([str(datapoint['id']), max(0,  datapoint['test'][0]['post_score'] - datapoint['test'][0]['pre_score']), datapoint['created_at']])

	children = np.array(children)
	children = pd.DataFrame(children, columns=['user_id', 'score', 'created_at'])
	children = children.sort_values('created_at')

	std1, std2, std3, std4, std5, std6 = 0,0,0,0,0,0

	for i in range(0, 1):
		# std1 += bootstrap(children.iloc[i:(i+5), :], 10000, 5)
		# std2 += bootstrap(children.iloc[i:(i+10), :], 10000, 10)
		# std3 += bootstrap(children.iloc[i:(i+15), :], 10000, 15)
		# std4 += bootstrap(children.iloc[i:(i+20), :], 10000, 20)
		std5 += bootstrap(children.iloc[i:(i + 25), :], 10000, 25)
		std6 += bootstrap(children.iloc[i:(i + 30), :], 10000, 30)

	std1, std2, std3, std4, std5, std6 = std1/30, std2/30, std3/30, std4/30, std5/30, std6/30
	a=1

def plotAllActions(data):
	histdic = {x: list(data).count(x) for x in data}

	keys = histdic.keys()
	values = histdic.values()

	plt.bar(keys, values)
	plt.title('All actions taken')
	plt.show()
	return 0

def plotActionTimeline(data, steps):
	hints = []
	nothings= []
	questions = []
	encouragements = []

	actions = list(data)
	print(len(actions))
	for i in range(1, len(actions)):
		if i <= steps:
			hints.append(actions[0:i].count('hint')/(i))
			nothings.append(actions[0:i].count('nothing') / (i ))
			questions.append(actions[0:i].count('question') / (i))
			encouragements.append(actions[0:i].count('encourage') / (i))
		else:
			hints.append(actions[(i-steps):i].count('hint') /steps)
			nothings.append(actions[(i-steps):i].count('nothing') / steps)
			questions.append(actions[(i-steps):i].count('question') / steps)
			encouragements.append(actions[(i-steps):i].count('encourage') / steps)

	ax = plt.subplot(111)
	ax.set_title('Average of last {} actions'.format(steps))
	ax.plot(hints, label='hints')
	ax.plot(nothings, label='nothing')
	ax.plot(encouragements, label='encourage')
	ax.plot(questions, label='question')
	ax.set_xlabel('Number of actions taken')
	ax.legend()
	plt.show()

def plotImprovements(filename, realChildrenFileName, steps, finishedChildrenFileName, childrenChars):
	with open(filename) as f:
		rewards = json.load(f)

	with open(childrenChars) as f2:
		childrenCharacteristics = json.load(f2)

	chars = {}
	for temp in childrenCharacteristics['data']['user']:
		chars[str(temp['id'])] = temp['grade']

	with open(finishedChildrenFileName) as f1:
		finishedChildren = json.load(f1)

	finishedChildrenList = []
	for temp in finishedChildren['data']['test']:
		if temp['adjusted_score'] != None:
			finishedChildrenList.append(temp['user_id'])

	realChildren = np.loadtxt(realChildrenFileName)
	adjusted_improvements = []
	plotableImprovements = []
	plotableAdjusted = []
	post_min_pre = []
	pre = []
	plotablePres = []
	grades = []
	plotableGrades = []

	LB = []
	UB = []
	LB2 = []
	UB2 = []

	interestingChildren = []

	i = 0
	for datapoint in rewards['data']['test']:
		if datapoint['user_id'] in realChildren and datapoint['user_id'] in finishedChildrenList:
			i += 1
			post_min_pre.append(datapoint['post_score'] - datapoint['pre_score'])
			pre.append(datapoint['pre_score'])
			grades.append(chars[str(datapoint['user_id'])])

			interestingChildren.append([i, datapoint['user_id'], datapoint['pre_score']])
			if datapoint['adjusted_score'] != None:
				adjusted_improvements.append(datapoint['adjusted_score'])
				begin = max(0, len(adjusted_improvements) - steps)
				plotableAdjusted.append(np.mean(adjusted_improvements[begin:]))
				CI = 1.96 * np.std(adjusted_improvements[begin:]) / np.sqrt(len(adjusted_improvements[begin:]))
				LB2.append(np.mean(adjusted_improvements[begin:]) - CI)
				UB2.append(np.mean(adjusted_improvements[begin:]) + CI)

			if len(post_min_pre) <= steps:
				plotableImprovements.append(np.mean(post_min_pre[0:]))
				plotablePres.append(np.mean(pre[0:]))
				plotableGrades.append(np.mean(grades[0:]))

				CI = 1.96 * np.std(post_min_pre[0:])/np.sqrt(len(post_min_pre[0:]))
				LB.append(np.mean(post_min_pre[0:])-CI)
				UB.append(np.mean(post_min_pre[0:])+CI)

			else:
				plotableImprovements.append(np.mean(post_min_pre[(len(post_min_pre)-steps):]))
				plotablePres.append(np.mean(pre[(len(pre)-steps):]))
				plotableGrades.append(np.mean(grades[(len(grades) - steps):]))

				CI = 1.96 * np.std(post_min_pre[(len(post_min_pre)-steps):]) / np.sqrt(steps)
				LB.append(np.mean(post_min_pre[(len(post_min_pre)-steps):]) - CI)
				UB.append(np.mean(post_min_pre[(len(post_min_pre)-steps):]) + CI)

	print('amount of children: {}'.format(len(plotableAdjusted)))
	# fig, ax = plt.subplots()
	# ax.plot(plotableAdjusted)
	# ax.fill_between(np.arange(0,len(plotableAdjusted)), LB2, UB2, color='b', alpha=0.1)
	# ax.set_title('Adjusted score averaged over last {} children'.format(steps))
	# ax.set_xlabel('number of children')
	# ax.set_ylabel('Adjusted score')
	# plt.show()

	fig, ax = plt.subplots()
	ax.plot(plotableGrades, label='grade')
	ax.legend()
	lines =  [5,  13, 25, 33, 47, 53, 65, 70, 77, 87, 103, 110, 119, 129]
	for line in lines:
		ax.axvline(x=line, color='r')
	ax.set_title('Grades over last {} children'.format(steps))
	ax.set_xlabel('number of children')
	ax.set_ylabel('Grade')
	plt.show()

	fig, ax = plt.subplots()
	ax.plot(plotableImprovements, label='Post-pre')
	ax.plot(plotablePres, label='Pre scores')
	lines = [5, 13, 25, 33, 47, 53, 65, 70, 77, 87, 103, 110, 119, 129]
	for line in lines:
		ax.axvline(x=line, color='r')
	ax.legend()
	ax.fill_between(np.arange(0,len(plotableImprovements)), LB, UB, color='b', alpha=0.1)
	ax.set_title('Post-pre over last {} children'.format(steps))
	ax.set_xlabel('number of children')
	ax.set_ylabel('Post-pre score')
	plt.show()


def plotRewards(data, rewardsFileName, steps, realChildrenFileName, finishedChildrenFileName):

	data = data[data[:,2].argsort()]
	realChildren = np.loadtxt(realChildrenFileName)
	with open(finishedChildrenFileName) as f1:
		finishedChildren = json.load(f1)

	finishedChildrenList = []
	for temp in finishedChildren['data']['test']:
		if temp['adjusted_score'] != None:
			finishedChildrenList.append(temp['user_id'])

	#get improvementdata
	with open(rewardsFileName) as f:
		improvementsData = json.load(f)

	dataImp = []
	for datapoint in improvementsData['data']['test']:
		if datapoint['user_id'] in realChildren and datapoint['user_id'] in finishedChildrenList:
			dataImp.append([datapoint['pre_score'], datapoint['post_score'], datapoint['adjusted_score'], datapoint['user_id']])

	improvementsData = np.array(dataImp)

	actions = list(data[:, 0])

	rewards = []
	plotableRewards = []
	currentID = data[0, 2]
	UB = []
	LB = []

	for i in range(0, len(actions)):
		if currentID != data[i, 2]:
			if improvementsData[:,2][improvementsData[:,3] == currentID][0] != None:
				rewards.append(improvementsData[improvementsData[:,3] == currentID, 2][0]) #append the final adjusted score reward for this specific id
			else:
				rewards.append(0)
			currentID = data[i,2]
		if currentID == data[i, 2]:
			currentID = data[i, 2]
			if actions[i] == 'hint':
				if data[i,1]:
					rewards.append(0.11)
				else:
					rewards.append(0.01)
			else:
				if data[i,1]:
					rewards.append(0.1)
				else:
					rewards.append(0)

		begin = max(0, len(rewards)-steps)
		plotableRewards.append(np.mean(rewards[begin:]))

		CI = 1.96 * np.std(rewards[begin:]) / np.sqrt(len(rewards[begin:]))
		LB.append(np.mean(rewards[begin:]) - CI)
		UB.append(np.mean(rewards[begin:]) + CI)

	fig, ax = plt.subplots()
	ax.plot(plotableRewards)
	ax.fill_between(np.arange(0, len(plotableRewards)), LB, UB, color='b', alpha=0.1)
	ax.set_title('Rewards over last {} actions'.format(steps))
	ax.set_xlabel('Number of rewards received')
	ax.set_ylabel('Averaged reward')
	plt.show()

	# plt.plot(plotableRewards)
	# plt.title('Rewards averaged over last {} actions'.format(steps))
	# plt.xlabel('Number of rewards received')
	# plt.ylabel('Averaged reward')
	# plt.show()



def makeImprovementPlots(file_name):
	realChildren = pd.read_csv(file_name).iloc[0:338,:]#.dropna()

	#plot pre-score  grade relationship
	# ax = sns.barplot(x="Pre", y="ImprovementAdjusted", hue="Condition", data=realChildren)
	# plt.show()



	condition_D_children = realChildren[realChildren['Condition'] == "Online D"]
	condition_A_children = realChildren[realChildren['Condition'] == "Online A"]

	a = condition_A_children[condition_A_children['Pre'] == 0]

	D_low = condition_D_children[condition_D_children["Pre"] < 2]
	D_high = condition_D_children[condition_D_children["Pre"] >= 2]

	A_low = condition_A_children[condition_A_children["Pre"] < 2]
	A_high = condition_A_children[condition_A_children["Pre"] >= 2]

	whole_test = scipy.stats.ttest_ind(condition_D_children['ImprovementAdjusted'], condition_A_children['ImprovementAdjusted'], equal_var=False)
	low = scipy.stats.ttest_ind(D_low['ImprovementAdjusted'], A_low['ImprovementAdjusted'], equal_var=False)
	high = scipy.stats.ttest_ind(D_high['ImprovementAdjusted'], D_high['ImprovementAdjusted'], equal_var=False)


	general_cohen = cohen_d(condition_D_children['ImprovementAdjusted'], condition_A_children['ImprovementAdjusted'])
	low_cohen = cohen_d(D_low['ImprovementAdjusted'],A_low['ImprovementAdjusted'])
	high_cohen = cohen_d(D_high['ImprovementAdjusted'], A_high['ImprovementAdjusted'])

	a = np.mean(abs(D_high['ImprovementAdjusted']))
	b = np.mean(abs(D_low['ImprovementAdjusted']))

	c = np.mean(abs(A_high['ImprovementAdjusted']))
	d = np.mean(abs(A_low['ImprovementAdjusted']))

	print("Average learning improvement D high: {}".format(np.mean(realChildren)))
	bin_list = [0,1,2,3,4,5,6,7,8]
	# plt.hist(condition_D_children['Pre'], bins= bin_list)
	# plt.title("Condition D pre test scores")
	# plt.ylabel("Frequency")
	# plt.xlabel("pre test score")
	# plt.plot()

	plt.hist(condition_A_children['Pre'], bins=bin_list)
	plt.title("Condition A pre test scores")
	plt.ylabel("Frequency")
	plt.xlabel("pre test score")
	plt.plot()

	print("Average A:{}".format(np.mean(abs(condition_A_children['ImprovementAbsolute']))))
	print("Average D:{}".format(np.mean(abs(condition_D_children['ImprovementAbsolute']))))

	# nAverage = 30
	# plt.plot(getAverageList(condition_D_children['Pre'], nAverage), label='Pre Score')
	# plt.plot(getAverageList(condition_D_children['ImprovementAbsolute'], nAverage), label='Post-Pre Score')
	# plt.plot(getAverageList(condition_D_children['ImprovementAdjusted'], nAverage), label="Adjusted Improvement")
	#
	# plt.title('Pre, improvement and adjusted improvement averaged over last {} actions'.format(nAverage))
	# plt.xlabel('Children')
	# plt.ylabel('Average score')
	# plt.legend()
	# plt.plot()


	#scatterplot
	# plt.scatter(realChildren['Pre'], realChildren['ImprovementAbsolute'], )
	_conditions = ["Condition A", "Condition D"]

	# fg = sns.FacetGrid(data=realChildren, hue='Condition')
	# fg.map(plt.scatter, 'Pre', 'ImprovementAbsolute', alpha=0.3).add_legend()

	nAverage = 30
	plt.plot(getAverageList(condition_D_children['Engagement'], nAverage))

	plt.title('Pre, post and adjusted improvement averaged over last {} actions'.format(nAverage))
	plt.xlabel('Children')
	plt.ylabel('Average Engagement')
	plt.plot()


	return 0

if __name__ == '__main__':

	#change path to current directory
	hint2actionFileName = '/Users/williamsteenbergen/PycharmProjects/SmartPrimerFall/ExperimentAnalysis/Hint2Action.json'
	realChildrenFileName = '/Users/williamsteenbergen/PycharmProjects/SmartPrimerFall/ExperimentAnalysis/Final_analysis/rlbot_results.csv'
	finishedChildrenFileName = '/Users/williamsteenbergen/PycharmProjects/SmartPrimerFall/ExperimentAnalysis/Final_analysis/finishedChildren.json'
	childrenChars = '/Users/williamsteenbergen/PycharmProjects/SmartPrimerFall/ExperimentAnalysis/Final_analysis/childrenCharacteristics.json'
	actionsTakenFileName = '/Users/williamsteenbergen/PycharmProjects/SmartPrimerFall/ExperimentAnalysis/Final_analysis/actionData.json'
	rewardsFileName = '/Users/williamsteenbergen/PycharmProjects/SmartPrimerFall/ExperimentAnalysis/Final_analysis/improvements.json'
	childrenData = '/Users/williamsteenbergen/PycharmProjects/SmartPrimerFall/ExperimentAnalysis/Final_analysis/children.json'

	hint2action = processHint2Action(hint2actionFileName)

	# makeImprovementPlots(realChildrenFileName)

	data = getData(hint2action, realChildrenFileName, actionsTakenFileName, finishedChildrenFileName)

	getData2(childrenData, data, realChildrenFileName, finishedChildrenFileName)
	#getDataControl(childrenData, data, realChildrenFileName, finishedChildrenFileName)

	plotAllActions(data)
	plotActionTimeline(data, 50)

	plotImprovements(rewardsFileName, realChildrenFileName, 5, finishedChildrenFileName, childrenChars)
	plotRewards(data, rewardsFileName, 50, realChildrenFileName, finishedChildrenFileName)

a=1