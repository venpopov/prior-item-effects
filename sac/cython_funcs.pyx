def decay_power(S, T, t, d, priorB):
    # check what the 1+ efficiency is - its a scalar
    T1 = np.power(1+t+T.data, -d)
    S1 = S.data
    if S.format == 'lil':
        S1 = list(itertools.chain(*S1))
    T.data = S1 * T1
    newB =  T @ np.ones(T.shape[1])
    if type(priorB) == int:
        B = newB
    else:
        B = priorB + newB

    return(B)
    
    
    
from math import exp
import numpy as np

def rbf_network(double[:, :] X,  double[:] beta, double theta):

    cdef int N = X.shape[0]
    cdef int D = X.shape[1]
    cdef double[:] Y = np.zeros(N)
    cdef int i, j, d
    cdef double r = 0

    for i in range(N):
        for j in range(N):
            r = 0
            for d in range(D):
                r += (X[j, d] - X[i, d]) ** 2
            r = r**0.5
            Y[i] += beta[j] * exp(-(r * theta)**2)

    return Y