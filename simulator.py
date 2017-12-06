import numpy as np
import sys
import pdb
import utils
from matplotlib import pyplot as plt
from policies import *
from exactmethods import *

class game:
	def __init__(self, n, m, N, H, T, pi):
		self.n = n # length
		self.m = m # widths
		self.N = N # Number of players
		self.A = 5 # Number of actions for each player
		self.s = 1 # Just initialize the state
		self.T = T # the transtion model
		self.horizon = H
		self.pi = pi

		# Rewards
		self.r = utils.build_r(n,m)

	def reset(self, x0):
		""" Resets the game to a new state with starting positions"""
		self.s = utils.pos2state(x0, self.n, self.m, self.N)
		self.R = 0

		# Reset trajectory history
		self.shist = [x0[i,:] for i in range(self.N)]
		self.ahist = [np.array([]) for i in range(self.N)]
		self.time = 0
		self.endconditionmet = False

	def run(self):
		""" Assuming it has already been reset """
		while not self.endconditionmet and self.time < self.horizon:
			# Determine the action
			a = self.action()

			# Take a step and collect reward by moving to new state sp
			[r,sp] = self.takeStep(self.s,a)

			# Check if end conditions are met
			self.checkEnd()

			# Update game parameters
			self.s = sp
			self.time += 1

			# Add the positions to the trajectory history
			self.updateHist(sp,a)

		self.showTrajectory()

	def action(self):
		""" Returns action for each player for state s from policy pi """
		acts = np.zeros((self.N,)) # initialize the actions
		
		# Attacker policy
		acts[0] = self.pi[self.s]

		# Defenders policy
		lookahead = 1
		p = 1
		acts[1] = Naive_D(self.s, self.n, self.m, lookahead, acts[0], p)

		# Convert the vector of player actions to single value
		a = int(utils.act2vec(acts, self.A))
		return a

	def takeStep(self,s,a):
		s = int(s)

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
		sp = int(nextstates[idx])

		# Compute reward
		r = self.r[sp]

		return r, sp
		
	def updateHist(self,s,a):
		X = utils.state2pos(s, self.n, self.m, self.N)
		acts = utils.vec2act(a,self.A,self.N)
		for i in range(self.N):
			x = np.matrix(X[i,:])
			self.shist[i] = np.append(self.shist[i], x, axis=0)
			self.ahist[i] = np.append(self.ahist[i], np.array([acts[i]]), axis=0)

	def checkEnd(self):
		""" Check if one of the end conditions is satisfied, (1) 
		Player 1 reaches the end of the grid, (2) Player 1 reaches a
		sideline, (3) time horizon is achieved """

		x = self.shist[0][-1,0] # Player 1 x position
		y = self.shist[0][-1,1]	# Player 1 y position

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
			plt.plot(self.shist[i][:,0], self.shist[i][:,1],
					marker='o')
		plt.show()



if __name__ == '__main__':
	n = 5 # Field length
	m = 5 # Field width
	N = 2 # Number of players
	H = 10 # Time horizon
	A = 5 # Number of actions

	print 'Computing transition probabilities ...'
	T  = utils.T(n,m,N,5,[1.0,0.8])

	print 'Computing policy using Value Iteration'
	[pi, U] = finiteHorizonValueIteration(H,n,m,A,T)

	sim = game(n,m,N,H,T,pi)
	x0 = np.matrix([[1,0],
					[1,1]])
	sim.reset(x0)
	sim.run()
	pdb.set_trace()
