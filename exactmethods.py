import numpy as np
import sys
import pdb
import utils
from policies import *


def finiteHorizonValueIteration(H,n,m,A,T):
	# Value iteration method, note this is built around having one attacker
	# and one defender only, no discounting, finite horizon

	# Size of the state space
	S = (n*m)**2

	# First determine all states that are absorbing states based on gameplay
	# and create the utility vector
	nabsorbing = (2*n + m - 2)*n*m + (n-1)*(m-2)

	sa = np.zeros((nabsorbing,), dtype=np.int)
	sc = np.zeros((S - nabsorbing,), dtype=np.int)
	U = np.zeros((S,)) # utility vector of length S
	pi = np.zeros((S,), dtype=np.int)

	idxabsorb = 0
	idxcont = 0
	for s in range(S):
		# Convert to positions
		pos = utils.state2pos(s,n,m,2)

		# Check if conditions on end game are met
		if pos[0,0] == 0 or pos[0,0] == m - 1 or pos[0,1] == n - 1 or \
				(pos[0,0] == pos[1,0] and pos[0,1] == pos[1,1]):
			sa[idxabsorb] = s
			U[s] = pos[0,1] # Utility of absorbing state is distance down field
			idxabsorb += 1

		# Otherwise the state s is a continuing state
		else:
			sc[idxcont] = s
			idxcont += 1

	# Now we have sa absorbing states and sc continuing states and Utilities
	# at absorbing states

	# Bellman update over time horizon
	Uold = np.copy(U)
	for k in range(H):
		for s in sc: # Only update states that are not absorbing
			# Now want to loop through each action to find the one that maximizes
			Q = np.zeros((A,))
			for a in range(A):
				sp = T[s,a,:,0].astype(np.int).tolist() # vector of states to go to
				t = T[s,a,:,1] # probability of transition
				Uvec = Uold[sp] # evaluate Uold at possible states sp
				Q[a] = np.dot(t,Uvec) # sum over all possible
			
			# Now choose the maximum Q
			U[s] = np.max(Q)
			pi[s] = np.argmax(Q)

	return pi, U


if __name__ == '__main__':
	H = 5 
	n = 7 # field length (n >= 2)
	m = 5 # field width (m >= 3)
	A = 5 # number of possible actions
	p = [0.9, 0.9] # probabilities of transitioning
	T = utils.T(n,m,2,A,p); # the transtion model
	[pi,U] = finiteHorizonValueIteration(H,n,m,A,T)
	pdb.set_trace()