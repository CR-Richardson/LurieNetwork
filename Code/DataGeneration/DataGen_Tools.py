import numpy as np
import scipy as sp
from scipy import optimize
from scipy.special import expit
import matplotlib.pyplot as plt



"""
A set of functions for generating the synthetic data.
"""



def SimpleAttractor(x0:np.array, time:np.array):
    """
    Function for generating trajectories of the simple attractor example.
    args:
           x0: Initial conditions (N, n) = (number of trajectories, dimension of state).
         time: Array containing the initial, final times of the simulation and the integration step size.
    returns:
            X: Trajectories (len(T), N, n).  
    """
    
    N, n = x0.shape
    t0 = time[0]; tmax = time[1]; step = time[2]
    T = np.arange(t0, tmax+step, step)

    A = np.array([[0, 1, -2], 
                  [-1, 0, -1], 
                  [0.5, 0, -0.5]])

    B = np.zeros((n,n)); B[2,0] = -0.5;
    
    C = np.zeros((n,n)); C[2,0] = 1
    
    X = np.zeros( (len(T),n,N), dtype=np.double )
    X[0] = np.transpose(x0) # n, N
    
    for i in range( len(T)-1 ):
        Y = np.matmul( C, X[i])
        X[i+1] = X[i] + step * ( np.matmul(A, X[i]) + np.matmul( B, Y**3 ) ) # tmax, n, N

    return np.transpose(X,(0,2,1) )



def Examples(name:str, x0:np.array, time:np.array):
    """ 
    Function for generating trajectories from the Hopfield Network and Opinion dynamics examples.
    args:
         name: 'Hopfield' or 'Opinion'.
           x0: Initial conditions (N, n) = (number of trajectories, dimension of state).
         time: Array containing the initial, final times of the simulation and the integration step size.
    returns:
            X: Trajectories (len(T), N, n).  
    """

    N, n = x0.shape
    t0 = time[0]; tmax = time[1]; step = time[2]
    T = np.arange(t0, tmax+step, step)

    X = np.zeros( (len(T),n,N), dtype=np.double )
    X[0] = np.transpose(x0) # n, N

        
    if name == 'Hopfield':
        alpha = 2.5
        A = -alpha*np.eye(3)

        B = 1.0*np.array([[1, 1, 1], 
                          [1, 1, 1], 
                          [1, 1, 1]])
            
        C = np.eye(3)

        b = np.zeros((3,))

        eqs = np.array([ [0.7903, 0.7903, 0.7903],
                         [0.0, 0.0, 0.0],
                         [-0.7903, -0.7903, -0.7903]
                      ])

        
    elif name == 'Opinion':
        A = -1.5*np.eye(3)

        B = 0.5*np.eye(3)
            
        C = np.array([ [1.0, -1.0, 0.0], 
                        [-1.0, 1.0, -1.0], 
                        [0.0, -1.0, 1.0]])

        b = np.array([0.2, 0.0, -0.2])

        eqs = b

    for i in range( len(T)-1 ):
        Y = np.matmul( C, X[i])
        X[i+1] = X[i] + step * ( np.matmul(A, X[i]) + np.matmul( B, np.tanh(Y) ) + np.tile(b, (N,1) ).T ) # tmax, n, N

    return np.transpose(X,(0,2,1) ), eqs


