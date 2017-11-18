import numpy as np
import sys
import pdb
import utils

class game:
	def __init__(self, n, m, N):
		self.n = n # length
		self.m = m # width
		self.N = N # Number of players
		self.A = 5 # Number of actions for each player
		self.s = 1 # Just initialize the state
		# self.pi = policy
		self.T = utils.T(n,m,N,5,[0.9,0.9]); # the transtion model
		
	def reset(self, x0):
		""" Resets the game to a new state with starting positions"""
		self.s = utils.pos2state(x0, self.n, self.m, self.N)
		self.R = 0

	def action(self, s):
		""" Returns action for each player for state s from policy pi """
		acts = np.zeros((self.N,)) # initialize the actions
		for i in range(self.N):
			# Temporarily, until policy is created
			acts[i] = 2 # go down the field (decrease y coordinate)

		# Convert the vector of player actions to single value
		a = int(utils.act2vec(acts, self.A))
		return a

	def update(self):
		# Determine the action based on each players policies and currents state
		s = self.s
		a = self.action(s)

		# For a given state and action pair, we have a distribution
		# over next states sp given by the transition model T
		nextstates = self.T[s,a,:,0]
		t = self.T[s,a,:,1]
		# vector = np.array([.1,.2,.3,.4])

		# Randomly select outcome based on probabilities
		randval = np.random.rand()
		for idx, p in enumerate(np.cumsum(t)):
			if randval <= p:
				print 'Outcome {}'.format(idx)
				break

		sp = nextstates[idx]
		self.s = sp
		pdb.set_trace()



	def show(self):
		print self.n


this = game(3,3,2)
x0 = np.matrix([[1,0],
				[1,1]])
this.reset(x0)

this.update()
pdb.set_trace()