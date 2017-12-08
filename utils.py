#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as scl

def T(n,m,N,A,p):
    # ReadMe
        # functionality
            # Right now, the number of actions A is 5.
            # Action 0 is remain in the same place.
            # Action 1 is go up (reduce y coordinate by 1)
            # Action 2 is go down (increase y coordinate by 1)
            # Action 3 is go left (decrease x coordinate by 1)
            # Action 4 is go right (increase x coordinate by 1)
        # INPUTS
            # n is the number of rows in our grid
            # m is the number of columns in our grid
            # N is the number of players on the field
            # A is the total number of actions possible for a single player.
            # p is a vector of size N whose entries are all in [0,1] which specify how noisy each of the players' transitions are.
                # for example, if player i chooses to go up, then with probability p[i] he will go up. He takes all other actions with equal probability.
        # OUTPUTS
            # A matrix, of size (n*m)^N by A^N by A^N by 2
                # the first 2 indices indicate state and actions.
                # The last 2 indices indicate possible future states and their probabilities

    L = n*m # number of locations on the field.

    T = np.zeros(((n*m)**N, A**N, A**N, 2))

    for s in range(L**N): # loop over all states
        pos = state2pos(s,n,m,N)
        for a in range(A**N): # loop over all actions
            acts = vec2act(a,A,N)
            for i in range(A**N): # loop over all possible realizations of the randomness
                mov = vec2act(i,A,N)
                t = 1 # initialize the probability of this realization to 1
                for j in range(N):
                    if (mov[j] != acts[j]):
                        t = t*(1-p[j])/float(4)
                    else:
                        t = t*p[j]
                new_pos,bounds = mov2pos(mov,pos,n,m)
                sp = pos2state(new_pos,n,m,N)
                T[s,a,i,0] = sp # record a possible state to transition to
                T[s,a,i,1] = t # record the probability of landing in this state

    return T

def state2pos(s,n,m,N):
    #ReadMe
        # functionality
            # takes one of (n^m)^N states and produces the positions of the N players
            # Our indices are ZERO INDEXED
            # We are using row major indexing.
            # REMARK: Error detection is currently not implemented.
        # INPUTS
            # N is the number of players in the game.
            # s is the state that we want to transform into N x,y pairs
            # n,m is the size of the field. n rows, m columns.
        # OUTPUTS
            # We output a N by 2 matrix which contains the positions of all N players.
            # Each row is the x,y coordinates of a player.

    pos = np.zeros((N,2)) # this is the output, will contain xy positions for all N players

    L = n*m # total number of locations in the grid
    for i in range(N):
        ith_player_state = np.mod(s, L) # get the piece of the state that encodes the x,y position of player i
        x,y = vec2mat(ith_player_state,n,m) # convert that substate into an xy position
        pos[i,0] = x #write the position of player i into the output
        pos[i,1] = y

        s = s - ith_player_state # remove player i's information from the state so that it will not interfere with later indicies
        s = s/L

    return pos

def pos2state(pos,n,m,N):
    #ReadMe
        #functionality
            # Takes the x-y positions of N players on a n by m grid and vectorizes that into a single integer state.
            # REMARK: Error detection is currently not implemented.
        #INPUTS
            # N is the number of players on the field
            # pos is a N by 2 matrix containing the x,y positions of all N players
            # The field/matrix/grid has n rows and m columns.
        #OUTPUTS
            # s is the row-major index that corresponds to pos. It takes values from 0 to (n*m)^N - 1

    L = n*m # total positions in the grid

    s = 0 # initiallize the state to zero
    for i in range(N):
        s = s + mat2vec(pos[i,0], pos[i,1], n,m)*(L**i) # add player i's position information to the state
    return s

def act2vec(acts,A):
    #ReadMe
        # functionality
            # takes the N actions of a player and converts it into a single action for the transition model T
            # Indices are ZERO indexed
        # INPUTS
            # acts is a list of N actions, one for each player. Has shape N by 1.
            # A is the total number of unique actions for each player.
        # OUTPUTS
            # a, which is a vectorized version of acts.

    N = acts.shape[0] # determine the total number of players.

    a = 0; #initialize the action
    for i in range(N):
        a = a + acts[i]*A**i # include the ith player's action information into a

    return a

def vec2act(a,A,N):
    #ReadMe
        # functionality
            # takes the compact MDP encoding of a set of actions and converts it into N actions, one for each player.
            # Indices are ZERO indexed
        # INPUTS
            # a describes the actions of all players. It is an integer, in the MDP format.
            # A is the total number of unique actions for each player.
            # N is the number of players in the game.
        # OUTPUTS
            # acts, which is a N length vector specifying the actions of each of the players.

    acts = np.zeros((N,)) # initialize the actions to output

    for i in range(N):
        acts[i] = np.mod(a, A) # extract the ith player's action
        a = a - acts[i] # remove the ith player's information from a so it will not interfere with later loop iterations
        a = a/A

    return acts

def mov2pos(mov,pos, n,m):
    #ReadMe
        # Given a set of movements, and a current set of positions, calculate the new positions of the N players.
    # INPUTS
        # mov is a vector of length N, which holds the displacement directions for all N players.
        # pos is a N by 2 matrix, where each row holds the xy coordinates of a player.
    # OUTPUTS
        # new_pos is the new position after the movement specified by mov is applied
        # bounds is a vector of length N, specifying if any of the players stepped out of bounds.
            # bounds[i] = 1 means the player is still in bounds after the movement, bounds[i] = 0 if the player moved out of bounds.

    N = mov.shape[0] # get the number of players
    new_pos = np.zeros((N,2))
    bounds = np.ones((N,))

    for i in range(N):
        if (mov[i] == 0): # no movement
            new_pos[i,0] = pos[i,0]
            new_pos[i,1] = pos[i,1]
        elif (mov[i] == 1): # go up
            new_pos[i,0] = pos[i,0]
            if (pos[i,1] > 0):
                new_pos[i,1] = pos[i,1] - 1
            else:
                new_pos[i,1] = pos[i,1]
                bounds[i] = 0

        elif (mov[i] == 2): # go down
            new_pos[i,0] = pos[i,0]
            if (pos[i,1] < n-1):
                new_pos[i,1] = pos[i,1] + 1
            else:
                new_pos[i,1] = pos[i,1]
                bounds[i] = 0
        elif (mov[i] == 3): # go left
            if (pos[i,0] > 0):
                new_pos[i,0] = pos[i,0] -1
            else:
                new_pos[i,0] = pos[i,0]
                bounds[i] = 0
            new_pos[i,1] = pos[i,1]
        elif (mov[i] == 4):
            if (pos[i,0] < m-1):
                new_pos[i,0] = pos[i,0] +1
            else:
                new_pos[i,0] = pos[i,0]
                bounds[i] = 0
            new_pos[i,1] = pos[i,1]
        else:
            print 'ERROR: Invalid movement.'

    return new_pos, bounds

def vec2mat(s,n,m):
    #ReadMe
        # functionality
            # convert a single index of a vector to 2 indices for a matrix
            # our indices are ZERO INDEXED.
            # We are using row major indexing.
        # INPUTS
            # s is the index of a vector we are converting to 2 indices in a matrix
            # n,m is the shape of the matrix
        # OUTPUTS
            # x,y are the indices in the matrix that correspond to s.
            # Returns -1 if there was an error.

    if ( (s < n*m) and (s >= 0) ): # check to see if s is a valid index
        x = np.mod(s,m) # compute the column position
        y = (s - x)/m # compute the row position
    else:
        print 'vec2mat ERROR: State out of bounds.'
        x = -1 # if s was not valid, return -1 for both x,y and print an error
        y = -1

    return x,y

def mat2vec(x,y,n,m):
    #ReadMe
        #functionality
            # convert 2 indices for a matrix to a single index for a long vector
            # our indices are ZERO INDEXED
            # We are using row major indexing.
        #INPUTS
            # x,y are the indices in the matrix whose corresponding vectorized index we wish to find.
            # n,m is the shape of the matrix.
        #OUTPUTS
            # s is the vectorized index in a n*m length vector corresponding to x,y in the row major mapping.

    if ( (0 <= x) and (x < m) and (0 <= y) and (y < n)): # if x,y are valid indices
        s = y*m + x # compute row major index mapping
    else:
        print 'mat2vec ERROR: x-y positions out of bounds'
        s = -1 # if x,y are not valid, print an error and return -1
    return s

def build_r(n,m):
    #ReadMe
        # functionality
            # Builds a vector whose length is the state size.
            # Contains the reward of going to that state
            # Zero for all non-terminal states.
            # This currently only works for N = 2, where there is one attacker and one defender
        # INPUTS:
            # n,m is the size of the field, rows and columns respectively.
        # OUTPUTS:
            # r is the terminal reward vector described above

    L = n*m

    r = np.zeros((L**2,))

    for s in range(L**2):
        pos = state2pos(s,n,m,2) # attacker is assumed to be player 1, i.e. the first row of pos is his position on the field
        # check if the attacker is out of bounds or tackled
        if ((pos[0,0] == 0) or (pos[0,1] == n-1) or (pos[0,0] == m-1) or ( (pos[0,0] == pos[1,0]) and (pos[0,1] == pos[1,1]) ) ):
            r[s] = pos[0,1] # in this case, his vertical advance is his reward

    return r

def build_R(T,r, n, m):
    # ReadMe
        # functionality
            # Build the function R(s,a) that specifies the *average* reward obtained by taking action a in state s.
            # Here we assume the number of players N is 2 and A is 5 (since this is the only setting for which r is defined)
        # INPUTS
            # T specifies the transition probabilities.
            # r specifies how valuable each destination is.
            # n,m is the number of rows and columns that the field contains.
        # OUTPUTS
            # R is a matrix with size num_states by num_actions which specifies the average reward for all state action pairs.

    A = 5 # number of actions each player can take
    N = 2 # number of players (attacker, defender)

    L = n*m # total locations

    R = np.zeros((L**N, A**N))

    for s in range(L**N):
        for a in range(A**N):
            temp = T[s,a,:,:]
            k = temp.shape[0]
            for i in range(k):
                R[s,a] = R[s,a] + temp[i,1]*r[int(temp[i,0])]

    return R

def load_policy(filename):
    # ReadMe:
        # functionality
            # loads a policy (a vector) from disc.
        # INPUTS
            # filename = 'something.csv'

    pi = np.genfromtxt(filename, delimiter = ',')
    # to save in the corresponding format, use:
    # np.savetxt(filename, variable, delimiter = ',')
    return pi

if __name__ == '__main__':
    # just some random junk code here to test the functionality of our utilities
    # pos = np.zeros((3,2))
    # pos[0,0] = 6
    # pos[0,1] = 8
    # pos[1,0] = 6
    # pos[1,1] = 1
    # pos[2,0] = 2
    # pos[2,1] = 2
    # s = pos2state(pos, 11,11,3)
    # print s
    # print state2pos(s,11,11,3)
    # N = 10
    # A = 7
    # acts = np.zeros((N,))
    # for i in range(N):
    #     acts[i] = np.mod(i+8,A)
    #
    # print acts
    # a = act2vec(acts, A)
    # acts_modem = vec2act(a, A, N)
    # print acts_modem

    n = 5
    m = 5
    N = 2

    num_s = (n*m)**N

    x = np.random.normal(0,1,(num_s,))
    filename = 'dummy_policy.csv'
    np.savetxt(filename, x, delimiter = ',')
    y = load_policy(filename)
    print np.linalg.norm(x-y)


    r = build_r(5,5)
