#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import utils

def Q_Learning(T,R):
    # ReadMe
        # functionality
            # performs Q-learning to train the attacker versus a static defender policy.
            # this function is only designed to work for N = 2, A = 5 (num_players, num_actions respectively)
        # INPUTS
            # The training data should have the sequential form (s,a,r,s'), similar to project 2.
        # OUTPUTS
            # pol is the optimal policy for the learned Q values.
            # Q is the matrix of size num_states by num_actions specifying the quality of a particular state action pair.

    L = n*m
    N = 2
    A = 5

    num_iters = 5

    Q = np.zeros((L**N, A**N))

    for t in range(num_iters): # do this num_iters times
        print "Q Learning iteration %s/%s" % (t,num_iters)
        for s in range(L**N):
            for a_a in range(A): # loop over the attacker's action
                d_a = Naive_D(s,n,m,1, a_a, 1) # determine the defender's action from Naive_D
                acts = np.zeros((2,))
                acts[0] = a_a
                acts[1] = d_a
                a = int(utils.act2vec(acts, A)) # get the full action

                temp = T[s,a,:,:] # find the transition information from T
                k = temp.shape[0]
                new_Q_val = R[s,a] # include the immediate reward

                for i in range(k): # for each possible next state s', we need to find max_{a'} Q(s',a')
                    sp = int(temp[i,0]) # this is one possible next state

                    max_Q = -1

                    for next_a_a in range(A): # loop over a' to find max_{a'} Q(s',a')
                        next_d_a = Naive_D(sp, n, m, 1, next_a_a, 1)
                        next_acts = np.zeros((2,))
                        next_acts[0] = next_a_a
                        next_acts[1] = next_d_a
                        next_a = int(utils.act2vec(next_acts, A)) # get the full next action

                        if (Q[sp, next_a] > max_Q):
                            max_Q = Q[sp, next_a]

                    new_Q_val = new_Q_val + temp[i,1]*max_Q

                Q[s,a] = new_Q_val

    return Q

def Naive_D(s, n, m, lookahead, a, p):
    # ReadMe
        # functionality
            # This is a policy that determines how the defender should act given a state s
            # Here we assume that the attacker is player 1 and the defender is player 2.
            # This is a simple policy that only works for N = 2, A = 5
            # With this policy, the defender aims to minimize horizontal distance first, to put himself between the attacker and the goal
            # This policy can allow the defender to "look ahead", by getting a noisy observation of the attacker's next step.
        # INPUTS
            # s is the state of the game which encodes the positions of both players. The attacker should be player 1. Defender is player 2
            # The field is a grid with n rows and m columns
            # lookahead is a boolean. If it is 1, then the defender gets a noisy observation of the attacker's next move.
            # a is the attacker's true action for this time step.
            # p is the probability that the defender's noisy observation is correct.
                # For example, if p = 0.6, then there is a 60% chance the defender's noisy observation is correct.
                # the other 4 outcomes are equally likely, so in this case they happen with proability 0.1 each.
        # OUTPUTS
            # a_d is the defender's action. It is an integer less than 5.

    if (lookahead == 1): # if we allow the defender to look ahead.
        # we will create the noisy observation via a state transition where the attacker
        # performs action a and the defender does not move.
        P = np.zeros((5,))
        for i in range(5): # create a probability vector of the defender's observation/measurement
            if (i == int(a)):
                P[i] = p # assign a with probability p
            else:
                P[i] = (1-p)/float(4) # assign equal probability to each of the incorrect observations

        X = np.random.multinomial(1, P) # draw from P
        a_obs = np.argmax(X) # get the noisy observation

        mov = np.zeros((2,))
        mov[0] = a_obs
        mov[1] = 0

        pre_pos = utils.state2pos(s, n, m, 2) # get the current position
        pos, bounds = utils.mov2pos(mov, pre_pos, n,m) # generate the new position. The attacker is player 1, the defender is player 2
    else: # if we do not allow lookahead,
        pos = utils.state2pos(s, n, m, 2)

    # now make a decision based on pos
    if (pos[0,0] > pos[1,0]): # if the attacker is to the right
        return 4
    elif (pos[0,0] < pos[1,0]): # if attacker is to the left
        return 3
    elif (pos[0,1] > pos[1,1]): # if the attacker is behind you
        return 2
    elif (pos[0,1] < pos[1,1]): # if the attacker is in front
        return 1
    else: # you are in the same spot as him.
        if (lookahead == 1):
            return 0 # then you should just stay
        else:
            return 2 # in this case, the attacker will try to advance, so you should follow him.



if __name__ == '__main__':
    # just some test code to see if the Naive_D function is working properly.
        n = 5
        m = 5
        N = 2
        A = 5

        p = np.zeros((2,))
        p[0] = 0.8
        p[1] = 0.9

        print "Building T..."
        T = utils.T(n,m,N,A,p)
        print "Building r..."
        r = utils.build_r(n,m)
        print "Building R..."
        R = utils.build_R(T,r,n,m)
        print "Running Q Learning..."
        Q = Q_Learning(T,R)
