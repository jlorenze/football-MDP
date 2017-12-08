#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from utils import *
from policies import *
from simulator import *

if __name__ == '__main__':

    SARSA_L_Pi = load_policy('SARSA_L_0.35_0.9_0.9_100000_0.05.csv')
    VI_L_Pi = load_policy('Value_Iteration.csv')

    n = 7
    m = 5
    N = 2
    A = 5
    H = 10
    p = np.zeros((2,))
    p[0] = 1
    p[1] = 0.8

    T = T(n,m,N,A,p)

    SL = game(n,m,N,H,T, SARSA_L_Pi) #instantiate game with Sarsa Lambda policy
    VI = game(n,m,N,H,T,VI_L_Pi) #instantiate game with Value iteration policy

    R_SL = 0 # avg reward for Sarsa Lambda
    R_VI = 0 # avg reward for Value Iteration

    tot_iter = 10000
    start = np.matrix( # specify the start state
        [[2,0],
         [2,1]]
    )

    for t in range(tot_iter):
        if (np.mod(t,50) == 0):
            print 'progress %s/%s' % (t,tot_iter)
        SL.reset(start)
        SL.run()
        R_SL = R_SL + SL.R/float(tot_iter)
        VI.reset(start)
        VI.run()
        R_VI = R_VI + VI.R/float(tot_iter)

    print 'Sarsa Lambda average reward: %s' % R_SL
    print 'Value Iteration average reward: %s' % R_VI
