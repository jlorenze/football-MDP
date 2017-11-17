#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as scl

def T(n,m,N,p,c):
    # ReadMe
    # INPUTS
        # n is the number of rows in our grid
        # m is the number of columns in our grid
        # N is the number of players on the field
        # p is between 0 and 1 and is the probability that the defender takes his
        # "best" action. He takes the other 4 actions with equal probability.
        # c < 1 is the parameter that makes the defender favor horizontal movement
        # to get between the running back and the goal
    # OUTPUTS
        # A matrix, of size (n*m)^N by 5 by (n,m)^N

    return 0


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
        ith_player_state = np.mod(np.mod(s, L**(i+1)), L**i) # get the piece of the state that encodes the x,y position of player i
        x,y = vec2mat(ith_player_state,n,m) # convert that substate into an xy position
        pos[i,0] = x #write the position of player i into the output
        pos[i,1] = y

        s = s - ith_player_state # remove player i's information from the state so that it will not interfere with later indicies

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


if __name__ == '__main__':
    # just some random junk code here to test the functionality of our utilities
    n = 5
    m = 5
    for i in range(n):
        for j in range(m):
            s = mat2vec(j,i,n,m)
            print s
