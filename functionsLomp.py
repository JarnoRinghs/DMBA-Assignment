import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gurobipy as gp
import random
import time
import math
np.random.seed(6)

#%% 6. Generate data
def GenerateKnapsackData(n, w_max):
    w = np.zeros(n)
    v = np.zeros(n)
    for i in range(n):
        w[i] = np.random.randint(low=1, high=w_max) #Make sure we do not have 0's. Divsision by 0 error
        v[i] = w[i] + np.random.randint(low=0, high=w_max//5)  #we let v[i] be correlated with w[i] with added randomness. 
    return w,v

def DynamicProgrammingKnapsack(n,w,v,W, return_M=False):
    M = {}
    for j in range(W+1):
        M[(0,j)] = 0
    for i in range(1,n+1):
        for j in range(W+1):
            if w[i-1] > j:
                M[(i,j)] = M[(i-1, j)]
            else:
                M[(i, j)] = max(M[(i - 1, j)], v[i-1] + M[(i-1, j - w[i-1])])
    opt = M[(n,W)]
    
    if return_M: 
        return M
    return opt
