#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import utils

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

    # print pos

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
    n = 10
    m = 10
    pos = np.zeros((2,2))
    pos[0,0] =  5 # x1
    pos[0,1] =  5 # y1
    pos[1,0] =  5 # x2
    pos[1,1] =  6 # y2

    s = utils.pos2state(pos,n,m,2)

    print Naive_D(s, n, m, 1, 3, 1)
