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
    alpha = 0.35 # learning rate, should be lower
    lam = 0.9
    gamma = 0.9
    num_iters = 10000
    tau = 0.05 # softmax parameter (tau = 0 means p(amax) = 1)

    print "Running SARSA lambda..."

    L = n*m # total number of locations
    N = 2 # number of players (attacker, defender)
    num_a = 5 # total number of unique actions
    num_s = L**N # total number of states

    # Initialize Q matrix
    Q = np.zeros((num_s, num_a)) # initialize the Q values to zero.

    # Initialize Q values for some states to nonzero values
    for s in range(num_s):
        pos = state2pos(s,n,m,2)
        if pos[0,1] >= pos[1,1] and (pos[0,0] != pos[1,0] or pos[0,1] != pos[1,1]):
            Q[s,2] = n

    picur = np.argmax(Q,axis=1)
    N = np.zeros((num_s, num_a)) # visit count for Sarsa_L

    # Start from several to increase chance of exploring
    start = [np.matrix([[2,0],[2,1]]),
             np.matrix([[1,0],[2,1]]),
             np.matrix([[3,0],[2,1]]),
             np.matrix([[2,0],[3,1]]),
             np.matrix([[2,0],[1,1]]),
             np.matrix([[1,0],[3,1]]),
             np.matrix([[3,0],[1,1]])]
    # start = [np.matrix([[2,0],[2,1]])]

    t = 0
    residual = np.inf
    norm = []
    U0hist = []
    pichange = []
    locs = np.zeros((n,m))
    while t < num_iters:
        print "progress %s/%s" % (t,num_iters)

        # Randomly select starting position
        idx = int(np.round((len(start)-1)*np.random.rand()))
        x0 = start[idx]
        sim.reset(x0) # initialize the game

        # Choose action a_0 with exploration
        P = np.exp(Q[sim.s,:]/tau)
        P = P/np.sum(P)
        a_a = np.argmax(np.random.multinomial(1, P)) # get the next action

        endflag = 0
        steps = 0
        Qold = np.copy(Q)
        piold = picur
        while (endflag == 0): # go until we reach a terminal state. Need to fix this to something else...
            # Observe reward and next state
            s = sim.s
            a = full_a[int(s), int(a_a)] # get the concatenated action (attacker, defender)
            r,sp = sim.takeStep(s, int(a)) # obtain the next state and reward
            sim.s = sp # update the state in the sim
            sim.checkEnd() # see if the game is over
            endflag = sim.endconditionmet

            # Check if captured
            if sim.capture and endflag:
                r = 0

            # Choose action a_t+1 with exploration
            P = np.exp(Q[sp,:]/tau)
            P = P/np.sum(P)
            next_a_a = np.argmax(np.random.multinomial(1, P)) # get the next action

            # Increment counts
            N[s,a_a] = N[s,a_a] + 1 # increment the counts
            
            # Update Q
            delta = r + gamma*Q[sp, next_a_a] - Q[s, a_a] # specify delta
            Q = Q + alpha*delta*N # update Q
            N = gamma*lam*N # update N

            # Update action and state for next iteration
            a_a = next_a_a
            steps += 1.0

        # Look at how much the policy changed
        picur = np.argmax(Q,axis=1)   
        diff = np.count_nonzero(picur - piold)/steps
        pichange.append(diff)
            
        # Look at how U[s0] is changing over time
        if t % 10 == 0:
            residual = np.linalg.norm(Q,ord='fro')
            expU0 = np.max(Q[s0])
            U0hist.append(expU0)
            norm.append(residual)

        # once the game finishes, we need to reset N
        N = np.zeros((num_s, num_a))
        t += 1

    # now that we have a Q function, we want to extract the optimal policy from this:
    pi_Q = np.zeros((num_s,))
    for s in range(num_s):
        pi_Q[s] = np.argmax(Q[s,:])

    # Now we would like to save the policy to disc
    filename = 'SARSA_L_%s_%s_%s_%s_%s.csv' % (alpha, lam, gamma, num_iters,tau)
    np.savetxt(filename,pi_Q,delimiter = ',')
    print 'Q density: %s' % (np.sum(Q)/float(num_s*num_a))

    # Lets do a moving average to smooth a bit
    num = len(U0hist)
    halfwidth = 20
    smooth = []
    for i,x in enumerate(U0hist):
        if i >= halfwidth and i <= num - 1 - halfwidth:
            smooth.append(np.mean(U0hist[i-halfwidth:i+halfwidth]))

    plt.plot(range(halfwidth, num - halfwidth),smooth)
    plt.show()

    # Look at the percentage of iterations where we don't change policy
    samples = 100.0
    num = len(pichange)
    percents = []
    idx = 0
    while samples*idx < num:
        zerocount = np.count_nonzero(pichange[100*idx:100*(idx+1)])
        percents.append(zerocount/samples)
        idx += 1

    plt.plot(percents,'b')
    plt.show()
    pdb.set_trace()
    return pi_Q,Q



if __name__ == '__main__':
    n = 7
    m = 5
    N = 2
    A = 5
    H = 10
    p = np.zeros((2,))
    p[0] = 1
    p[1] = 0.8

    # T = TTrain(n,m,N,A,p)
    T = T(n,m,N,A,p)
    full_a = Naive_Fullstate(n,m,1) # defender has perfect lookahead for 1 time step.

    sim = game(n,m, 2, H, T)
    pi,Q = Sarsa_L(n,m,full_a,sim)
