import numpy as np
import sys
import pdb

class game:
	def __init__(self, n, m):
		self.n = n #
		self.m = m #
		self.s = 1 # Just initialize the state
		# self.pi = policy
		# self.T = transitionModel(); # the transtion model
		
	def reset(self, A0, D0):
		""" Resets the game to a new state with starting positions"""
		self.s = pos2state(A0,D0)
		self.R = 0

	def action(self):
		""" Returns action for state s from policy pi """
		return self.pi[self.s]

	def update(self):
		# a = self.action()

		# For a given state and action pair, we have a distribution
		# over next states sp given by the transition model T
		# vector = T[vals]
		vector = np.array([.1,.2,.3,.4])

		# Lets assume that each value in vector is the ones for which
		# there is a nonzero probablity of transitioning
		probs = np.cumsum(vector)

		# Randomly select outcome based on probabilities
		randval = np.random.rand()
		for outcome, p in enumerate(probs):
			if randval <= p:
				print 'Outcome {}'.format(outcome)
				break

		sp = outcome
		self.s = sp

		# Check for collisions

		# Check if out of bounds




	def show(self):
		print self.n


this = game(10,10)
this.update()