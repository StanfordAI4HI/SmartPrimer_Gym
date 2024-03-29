import gym
from gym import spaces
from gym_SmartPrimer.envs.V2.Realistic.Childclass  import Child
import numpy as np
from gym_SmartPrimer.envs.V2.Realistic import ChildBehavior as ChildBehavior
import matplotlib.pyplot as plt
from gym_SmartPrimer.envs.V2.Realistic import NextObservation as nextObs
import json
import os

class SmartPrimerDynamicEnv(gym.Env):
	"""V2 smart primer environment"""

	metadata = {'render.modes': ['human']}

	def __init__(self):
		'''Initializes the environmenta'''
		with open(os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'childConfig.json'))) as config_file:
			self.settings = json.load(config_file)

		self.env = {}
		self.info = {}

		self.RewardsPerChild = []
		self.ImprovementPerChild = []
		self.performance = []
		self.improvement = []

		self.nFinish = []
		self.avgFinish = []

		self.Nquit = []
		self.avgQuit = []

		self.childrenSimulated = 0

		# pre-test, grade, age, seconds of last interaction, seconds of last correct answer,
		# [0,0,0] (positive, idk, negative) words since last action taken, stage of the problem,
		# seconds since last interaction with wizard, anxiety
		# low = np.array((0, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0), dtype = float)
		# high = np.array((8, 6, 10, 1000, 1000, 1, 1, 1, 3, 1000, 45), dtype = float) #pre-test, 4 words dim, 3 prev-hints

		# low = np.array((0, 2, 0, 0, 0, 0, 0, 0), dtype=float)
		# high = np.array((8, 6, 1, 1, 1, 3, 45, 10), dtype=float)  # pre-test, grade, 4 words dim, stage, anxiety ,number of wrong answers

		low = np.array((-1, -1, -1, -1, -1, -1, -1, -1), dtype=float)
		high = np.array((1, 1, 1, 1, 1, 1, 1, 1), dtype=float)

		self.observation_space = spaces.Box(low, high, dtype=np.float)

		self.action_space = spaces.Discrete(4)  #do nothing, encourage, ask question or provide hints
		self.actions = ['nothing', 'encourage', 'question',  'hint']
		self.reward_range = (-8, 9)

		self.actionInfo = {'encourage': [], 'question': [],  'nothing': [], 'hint': []}
		self.avgActionInfo = {'encourage': [], 'question': [], 'nothing': [], 'hint': []}

		kids = range(0, self.settings['nTypes'])
		for kid in kids:
			self.actionInfo[str(kid)] = [[], [], [], []]
			self.avgActionInfo[str(kid)] = [[], [], [], []]

		self.reset()

	def step(self, action):
		'''Takes an action and return the new statespace, reward, whether the episode has ended and some performance info'''
		self.getInfo(action)

		# the needed time for a child will decrease with one.
		self.child.neededTime -= 1
		action = self.actions[action]
		improvement, reward, done, info = ChildBehavior.react2action(action, self.child, self.stage, self.interactions)
		self.childRewards += reward

		if not done:
			self.state, self.interactions, self.stage = nextObs.nextObservation(self.child, self.interactions, action,
			                                                                    self.stage)  # first 30 secs

		if done:
			# print('This is the kids improvement: {}'.format(improvement))

			self.ImprovementPerChild.append(improvement)
			self.RewardsPerChild.append(self.childRewards)
			performance = self.RewardsPerChild[-min(len(self.RewardsPerChild)-1, 50):]
			self.performance.append(np.mean(performance))

			improvementSum = self.ImprovementPerChild[-min(len(self.ImprovementPerChild), 50):]
			# print('ImprovementSum: {}'.format(improvementSum))

			self.improvement.append(np.mean(improvementSum))

			nQuit, nFinish = 0, 0
			if info['reaction'] == 'quit':
				nQuit = 1
			elif info['reaction'] == 'finished':
				nFinish = 1

			self.Nquit.append(nQuit)
			avgQuit = np.mean(self.Nquit[-min(len(self.Nquit)-1, 50):])
			self.avgQuit.append(avgQuit)

			self.nFinish.append(nFinish)
			avgFinish = np.mean(self.nFinish[-min(len(self.nFinish)-1, 50):])
			self.avgFinish.append(avgFinish)

			# print('This is the average improvement seen so far: {}'.format(self.improvement))

		self.info = {'RewardsPerChild': self.RewardsPerChild, 'Performance': self.performance,
		             'Improvement': self.improvement,  'nFinish': self.avgFinish,
		             'nQuit': self.avgQuit, 'actionInfo': self.avgActionInfo, 'improvementPerChild': improvement, 'action': info['action']}

		return self.state, reward, done, self.info

	def getInfo(self, actionTaken):
		'''Retreives performance measures for plotting'''
		kids = range(0, self.settings['nTypes'])
		actions = [0, 1, 2, 3]

		#kid specific
		for kid in kids:
			for action in actions:
				if kid == self.child.type and action == actionTaken:
					self.actionInfo[str(self.child.type)][actionTaken].append(1)
					temp = self.actionInfo[str(self.child.type)][actionTaken]
					self.avgActionInfo[str(self.child.type)][actionTaken].append(np.mean(temp[-min(len(temp), 500):]))
				elif kid == self.child.type:
					self.actionInfo[str(kid)][action].append(0)
					temp = self.actionInfo[str(kid)][action]
					self.avgActionInfo[str(kid)][action].append(np.mean(temp[-min(len(temp), 500):]))

		#for all kids
		for action in actions:
			if action == actionTaken:
				self.actionInfo[self.actions[action]].append(1)
			else:
				self.actionInfo[self.actions[action]].append(0)

			self.avgActionInfo[self.actions[action]].append(np.mean(self.actionInfo[self.actions[action]][-min(len(self.actionInfo[self.actions[action]]), 500):]))
		return 0


	def reset(self):
		'''Starts a new episode by creating a new child and resetting performance, stage, observation space.'''
		self.childrenSimulated += 1
		if self.childrenSimulated % 50 == 0:
			print('We simulated {} children'.format(self.childrenSimulated))

		self.interactions = [0, 0, 0]
		self.child = Child(self.settings)  # create a child of random type
		self.childRewards = 0

		prevAction = 'nothing'
		self.stage = 0

		self.state, self.interactions, self.stage = nextObs.nextObservation(self.child, self.interactions, prevAction, self.stage) #first 30 secs
		return self.state

	def render(self, mode='human'):
		'''Creates plots of the results'''
		ax1 = plt.subplot(411)
		# ax1.set_title('Scenario 4: average reward of last 100 children without quitting penalty')
		ax1.margins(0.05)
		# ax1.set_xlabel('Number of children')
		ax1.set_title('Average reward of last 50 children')
		ax1.plot(self.info['Performance'], 'r')

		ax5 = plt.subplot(412)
		ax5.set_title('Average improvement of last 50 children')
		ax5.plot(self.info['Improvement'], 'r')

		ax4 = plt.subplot(413)
		ax4.set_title('Actions taken')
		ax4.plot(self.info['actionInfo']['question'], 'r', label='Question')
		ax4.plot(self.info['actionInfo']['encourage'], 'g', label='Encourage')
		ax4.plot(self.info['actionInfo']['hint'], 'b', label='Hint')
		ax4.plot(self.info['actionInfo']['nothing'], 'y', label='Nothing')
		ax4.set_ylabel('% of last 500 actions')
		ax4.legend()

		ax2 = plt.subplot(414)
		ax2.plot(self.info['nQuit'], 'r', label='Quit')
		ax2.plot(self.info['nFinish'], 'g', label='Finished')
		ax2.set_ylabel('% of last 100 children')
		ax2.legend()

		plt.show()

		nChildren = self.settings['nTypes']
		fig, axes = plt.subplots(nrows=1, ncols=nChildren)

		i = 0
		if nChildren != 1:
			for ax in axes.flatten():
				ax.set_title('Child type {}'.format(i+1))
				ax.plot(self.info['actionInfo'][str(i)][0], 'y', label='Nothing')
				ax.plot(self.info['actionInfo'][str(i)][1], 'g', label='Encourage')
				ax.plot(self.info['actionInfo'][str(i)][2], 'r', label='Question')
				ax.plot(self.info['actionInfo'][str(i)][3], 'b', label='Hint')
				ax.set_ylabel('% of last 500 actions')
				ax.legend()
				i=i+1
		else:
			axes.set_title('Child type {}'.format(i + 1))
			axes.plot(self.info['actionInfo'][str(i)][0], 'y', label='Nothing')
			axes.plot(self.info['actionInfo'][str(i)][1], 'g', label='Encourage')
			axes.plot(self.info['actionInfo'][str(i)][2], 'r', label='Question')
			axes.plot(self.info['actionInfo'][str(i)][3], 'b', label='Hint')
			axes.set_ylabel('% of last 500 actions')
			axes.legend()

		plt.show()


