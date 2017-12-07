#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from utils import *
from policies import *
from simulator import *
# also
# import matplotlib.pylab as plt

def Sarsa_L(n,m, full_a,sim):
    # SARSA LAMBDA parameters:
    alpha = 0.3 # learning rate, should be lower
    lam = 0.9
    gamma = 0.9
    num_iters = 10000

    sftmx = 0.5 # softmax parameter

    print "Running SARSA lambda..."

    L = n*m # total number of locations
    N = 2 # number of players (attacker, defender)
    num_a = 5 # total number of unique actions
    num_s = L**N # total number of states

    Q = np.zeros((num_s, num_a)) # initialize the Q values to zero.
    N = np.zeros((num_s, num_a)) # visit count for Sarsa_L

    start = np.matrix( # specify the start state
        [[2,0],
         [2,1]]
    )
    s0 = pos2state(start,n,m,2)
    a_a = 2

    t = 0
    residual = np.inf
    norm = []
    U0hist = []
    while t < num_iters:
        print "progress %s/%s" % (t,num_iters)
        sim.reset(start) # initialize the game
        endflag = 0
        while (endflag == 0): # go until we reach a terminal state. Need to fix this to something else...
            # Observe reward and next state
            a = full_a[int(sim.s), int(a_a)] # get the concatenated action (attacker, defender)
            r,sp = sim.takeStep(sim.s, int(a)) # obtain the next state and reward

            # Choose action a_t+1 with exploration
            normQ = np.linalg.norm(Q[sp, :])
            if normQ > 0:
                P = np.exp(Q[sp, :]/normQ)
            else:
                P = np.exp(Q[sp, :])
            
            P = P/np.sum(P)
            next_a_a = np.argmax(np.random.multinomial(1, P)) # get the next action

            # Increment counts
            N[sim.s,a_a] = N[sim.s,a_a] + 1 # increment the counts
            
            # Update Q
            delta = r + gamma*Q[sp, next_a_a] - Q[sim.s, a_a] # specify delta
            Q = Q + alpha*delta*N # update Q
            N = gamma*lam*N # update N

            # Update action and state for next iteration
            a_a = next_a_a
            sim.s = sp # update the state
            sim.checkEnd() # see if the game is over
            endflag = sim.endconditionmet

        # # Use Bellman residual to indiciate training process
        # residual = np.linalg.norm(Q-Qold,ord='fro') # max norm of Q upate
        if t % 10 == 0:
            residual = np.linalg.norm(Q,ord='fro')
            expU0 = np.max(Q[s0])
            if expU0 > 100:
                pdb.set_trace()
            U0hist.append(expU0)
            norm.append(residual)

        t += 1

        # once the game finishes, we need to reset N
        N = np.zeros((num_s, num_a))

    # now that we have a Q function, we want to extract the optimal policy from this:
    pi_Q = np.zeros((num_s,))

    for s in range(num_s):
        pi_Q[s] = np.argmax(Q[s,:])

    # now we would like to save the policy to disc
    filename = 'SARSA_L_%s_%s_%s_%s.csv' % (alpha, lam, gamma, num_iters)
    np.savetxt(filename,pi_Q,delimiter = ',')
    print 'Q density: %s' % (np.sum(Q)/float(num_s*num_a))

    # Lets do a moving average to smooth a bit
    num = len(U0hist)
    halfwidth = 20
    smooth = []
    for i,x in enumerate(U0hist):
        if i >= halfwidth and i <= num - 1 - halfwidth:
            smooth.append(np.mean(U0hist[i-halfwidth:i+halfwidth]))

    # plt.plot(range(num_iters),U0hist)
    plt.plot(range(halfwidth, num - halfwidth),smooth)
    plt.show()
    pdb.set_trace()

    return pi_Q,Q



if __name__ == '__main__':
    n = 5
    m = 5
    N = 2
    A = 5
    p = np.zeros((2,))
    p[0] = 1
    p[1] = 0.8

    T = T(n,m,N,A,p)
    full_a = Naive_Fullstate(n,m,1) # defender has perfect lookahead for 1 time step.

    sim = game(n,m, 2, 20, T)
    pi,Q = Sarsa_L(n,m, full_a,sim)
