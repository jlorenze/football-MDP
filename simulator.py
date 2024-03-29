import numpy as np
import sys
import pdb
import utils
from matplotlib import pyplot as plt
from policies import *
from exactmethods import *

class game:
	def __init__(self, n, m, N, H, T, pi=None, display=False):
		self.n = n # length
		self.m = m # widths
		self.N = N # Number of players
		self.A = 5 # Number of actions for each player
		self.s = 1 # Just initialize the state
		self.T = T # the transtion model
		self.horizon = H
		self.pi = pi
		self.display = display

		# Rewards
		self.r = utils.build_r(n,m)
		self.R = 0
		self.capture = False

	def reset(self, x0):
		""" Resets the game to a new state with starting positions"""
		self.s = utils.pos2state(x0, self.n, self.m, self.N)
		self.R = 0
		self.capture = False

		# Reset trajectory history
		self.shist = [x0[i,:] for i in range(self.N)]
		self.ahist = [np.array([]) for i in range(self.N)]
		self.time = 0
		self.endconditionmet = False

	def run(self):
		""" Assuming it has already been reset """
		if self.pi is None:
			print 'Needs a policy to run'
			sys.exit()

		while not self.endconditionmet and self.time < self.horizon:
			# Determine the action
			a = self.action()

			# Take a step and collect reward by moving to new state sp
			[r,sp] = self.takeStep(self.s,a)

			# Update game parameters
			self.s = sp
			self.time += 1

			# Check if end conditions are met
			self.checkEnd()

			# Add the positions to the trajectory history
			self.updateHist(sp,a)

		if self.display:
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

		pos = utils.state2pos(self.s, self.n, self.m, self.N)
		x = pos[0,0] # Player 1 x position
		y = pos[0,1] # Player 1 y position
		self.x = x
		self.y = y

		if (0 == x) or (x == self.m - 1):
			self.endconditionmet = True
			# print 'Sideline reached.'

		if y == self.n - 1:
			self.endconditionmet = True
			# print 'End of field reached.'

		if self.time == self.horizon:
			self.endconditionmet = True
			# print 'Time Horizon met.'

		# Check if the defender has captured the attacker
		if all(pos[0,:] == pos[1,:]):
			self.endconditionmet = True
			self.capture = True
			# print 'Offense was captured.'

		self.R = self.y
		

	def showTrajectory(self):
		plt.figure()
		ax = plt.gca()
		ax.set_ylim([-1, self.n-1])
		ax.set_xlim([0, self.m-1])
		ax.set_aspect('equal')
		ax.invert_yaxis()
		plt.plot(self.shist[0][:,0], self.shist[0][:,1],marker=None,markersize=10,color='r')
		plt.plot(self.shist[1][:,0], self.shist[1][:,1],marker=None,linestyle='--',color='b')
		plt.plot(self.shist[0][0,0], self.shist[0][0,1],marker='x',markersize=10,color='r')
		plt.plot(self.shist[1][0,0], self.shist[1][0,1],marker='o',linestyle='--')
		plt.plot(self.shist[0][-1,0], self.shist[0][-1,1],marker='x',markersize=10,color='r')
		plt.plot(self.shist[1][-1,0], self.shist[1][-1,1],marker='o',linestyle='--',color='b')
		plt.xlabel('Field Width')
		plt.ylabel('Distance Down Field')
		plt.title('Reinforcement Learning Policy 1')
		plt.show()



if __name__ == '__main__':
	n = 7 # Field length
	m = 5 # Field width
	N = 2 # Number of players
	H = 10 # Time horizon
	A = 5 # Number of actions

	print 'Computing transition probabilities ...'
	T  = utils.T(n,m,N,5,[1.0,0.8])

	print 'Computing policy using Value Iteration'
	# [pi, U] = finiteHorizonValueIteration(H,n,m,A,T)
	pi = load_policy('fullr_SARSA_L_0.3_0.9_0.9_10000_0.05.csv')

	sim = game(n,m,N,H,T,pi,display=True)
	x0 = np.matrix([[2,0],
					[2,1]])
	s0 = utils.pos2state(x0,n,m,N)

	sim.reset(x0)
	sim.run()
	pdb.set_trace()
