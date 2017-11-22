import numpy as np
import sys
import pdb
import utils
from matplotlib import pyplot as plt
from policies import *

class game:
	def __init__(self, n, m, N, H):
		self.n = n # length
		self.m = m # width
		self.N = N # Number of players
		self.A = 5 # Number of actions for each player
		self.s = 1 # Just initialize the state
		# self.pi = policy
		self.T = utils.T(n,m,N,5,[0.9,0.9]); # the transtion model
		self.horizon = H

	def reset(self, x0):
		""" Resets the game to a new state with starting positions"""
		self.s = utils.pos2state(x0, self.n, self.m, self.N)
		self.R = 0

		# Reset trajectory history
		self.hist = [x0[i,:] for i in range(self.N)]
		self.time = 0
		self.endconditionmet = False

	def run(self):
		""" Assuming it has already been reset """
		while not self.endconditionmet:
			self.update()
			self.checkEnd()

		self.showTrajectory()

	def action(self, s):
		""" Returns action for each player for state s from policy pi """
		acts = np.zeros((self.N,)) # initialize the actions
		
		acts[0] = 2

		# Defenders policy
		lookahead = 1
		p = 1
		acts[1] = Naive_D(s, self.n, self.m, lookahead, acts[0], p)

		# Convert the vector of player actions to single value
		a = int(utils.act2vec(acts, self.A))
		return a

	def update(self):
		# Determine the action based on each players policies and currents state
		s = int(self.s)
		a = self.action(s)

		# For a given state and action pair, we have a distribution
		# over next states sp given by the transition model T
		nextstates = self.T[s,a,:,0]

		t = self.T[s,a,:,1]

		# Randomly select outcome based on probabilities
		randval = np.random.rand()
		for idx, p in enumerate(np.cumsum(t)):
			if randval <= p:
				break

		# Update the next state
		sp = nextstates[idx]
		self.s = sp
		self.time += 1

		# Add the positions to the trajectory history
		self.updateHist(sp)

	def updateHist(self,s):
		X = utils.state2pos(s, self.n, self.m, self.N)
		for i in range(self.N):
			x = np.matrix(X[i,:])
			self.hist[i] = np.append(self.hist[i], x, axis=0)

	def checkEnd(self):
		""" Check if one of the end conditions is satisfied, (1) 
		Player 1 reaches the end of the grid, (2) Player 1 reaches a
		sideline, (3) time horizon is achieved """

		x = self.hist[0][-1,0] # Player 1 x position
		y = self.hist[0][-1,1]	# Player 1 y position

		if (0 == x) or (x == self.m - 1):
			self.endconditionmet = True
			print 'Sideline reached.'

		if y == self.n - 1:
			self.endconditionmet = True
			print 'End of field reached.'

		if self.time == self.horizon:
			self.endconditionmet = True
			print 'Time Horizon met.'

		# Check if the defender has captured the attacker
		pos = utils.state2pos(self.s, self.n, self.m, self.N)
		if all(pos[0,:] == pos[1,:]):
			self.endconditionmet = True
			print 'Offense was captured.'
		

	def showTrajectory(self):
		plt.figure()
		ax = plt.gca()
		ax.set_ylim([0, self.n-1])
		ax.set_xlim([0, self.m-1])
		ax.set_aspect('equal')
		ax.invert_yaxis()
		for i in range(self.N):
			plt.plot(self.hist[i][:,0], self.hist[i][:,1],
					marker='o')
		plt.show()



if __name__ == '__main__':
	n = 5 # Field length
	m = 5 # Field width
	N = 2 # Number of players
	H = 10 # Time horizon
	simulator = game(n,m,N,H)
	x0 = np.matrix([[1,0],
					[1,1]])
	simulator.reset(x0)
	simulator.run()
