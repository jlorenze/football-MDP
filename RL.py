#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from utils import *
from policies import *
from simulator import *
# also
# import matplotlib.pylab as plt

def Sarsa_L(n,m, full_a):
    # SARSA LAMBDA parameters:
    alpha = 0.8
    lam = 0.9
    gamma = 0.8
    num_iters = 10

    print "Running SARSA lambda..."

    sim = game(n,m, 2, 10)

    L = n*m # total number of locations
    N = 2 # number of players (attacker, defender)
    num_a = 5 # total number of unique actions
    num_s = L**N # total number of states

    Q = np.zeros((num_s, num_a)) # initialize the Q values to zero.
    N = np.zeros((num_s, num_a)) # visit count for Sarsa_L

    start = np.matrix( # specify the start state
        [[0,2],
        [2,2]]
    )

    for t in range(num_iters):
        print "progress %s/%s" % (t,num_iters)
        sim.reset(start) # initialize the game
        endflag = 0
        while (endflag == 0): # go until we reach a terminal state. Need to fix this to something else...
            # use softmax to pick an action:
            P = np.exp(Q[int(sim.s),:])
            P = P/np.sum(P) # extract the softmax distribution from Q
            a_a = np.argmax(np.random.multinomial(1, P)) # draw an action from P

            N[sim.s,a_a] = N[sim.s,a_a] + 1 # increment the counts
            a = full_a[int(sim.s), int(a_a)] # get the concatenated action (attacker, defender)

            r,sp = sim.takeStep(sim.s, a) # obtain the next state and reward
            P = np.exp(Q[int(sp), :])
            P = P/np.sum(P)
            next_a_a = np.argmax(np.random.multinomial(1, P)) # get the next action

            delta = r + gamma*Q[sp, next_a_a] - Q[sim.s, a_a] # specify delta
            Q = Q + alpha*delta*N # update Q
            N = gamma*lam*N # update N

            sim.s = sp # update the state
            sim.checkEnd() # see if the game is over
            endflag = sim.endconditionmet

        # once the game finishes, we need to reset N
        N = np.zeros((num_s, num_a))

    return Q



if __name__ == '__main__':
    n = 5
    m = 5
    N = 2
    A = 5
    p = np.zeros((2,))
    p[0] = 0.8
    p[1] = 0.9

    # T = T(n,m,N,A,p)
    # r = build_r(n,m)
    full_a = Naive_Fullstate(n,m,1) # defender has perfect lookahead for 1 time step.

    Q = Sarsa_L(n,m, full_a)

    # x = np.zeros((5,))
    # P = np.exp(x)
    # P = P/np.sum(P)
    # print P
    # print np.argmax(np.random.multinomial(1,P))
