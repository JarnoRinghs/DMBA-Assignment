#Implement correct packages
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gurobipy as gp
import random
import time
import math
from tensorflow import keras
from tensorflow.keras import layers


#%% 6. Generate data
def GenerateKnapsackData(n,w_max,alpha):
    weights = np.random.randint(1,w_max,n)
    profits = weights + np.ceil(w_max/10)
    W = int(round(alpha*np.sum(weights),0))
    return weights,profits,W
alpha = 0.4


#%% 1. Binary Programming of KP
def BinaryProgrammingKnapsack(n,W,v,w):
    knapsack = gp.Model("Knapsack Binary Programming")
    
    # Indices/
    index = [i for i in range(n)]
    
    # Parameters / Variables
    x = knapsack.addVars(index, vtype="B", name="x", lb=0)
    w_dict = {} #make dictionaries for Gurobi
    v_dict = {}
    for i,w in enumerate(w):
        w_dict[i] = w
    for i,v in enumerate(v):
        v_dict[i] = v
    
    # Constant is W
    
    # Objective Function
    knapsack.setObjective(sum(v_dict[i]*x[i] for i in index), gp.GRB.MAXIMIZE)
    
    # Constraints
    knapsack.addConstr(sum(w_dict[i]*x[i] for i in index)<=W)
    
    # Solve
    knapsack.optimize()
    opt_ind = [i for i in range(n) if x[i].x ==1]
    opt_sol = [x[i].x for i in range(n)]
    #print(opt_sol)
    opt_val = knapsack.ObjVal
    return opt_ind

 

#%% 2. Dynamic Programming of KP
#If we set n=10000, we get memory error after a while
def DynamicProgrammingKnapsack(n,W,v,w):
    M = {}
    for j in range(W+1):
        M[(0,j)] = 0
    for i in range(1,n+1):
        for j in range(W+1):
            if w[i-1] > j:
                M[(i,j)] = M[(i-1, j)]
            else:
                M[(i, j)] = max(M[(i - 1, j)], v[i-1] + M[(i-1, j - w[i-1])])
        #print('Point: {}.'.format(i))
    opt = M[(n,W)]
    opt_ind = []
    u = W
    #start backtracking
    for i in range(n,0,-1):
        if M[(i,u)] != M[(i-1, u)]:
            opt_ind.append(i-1)
            u -= w[i-1]
    opt_ind.reverse()
    return opt_ind

     
#%% 3. Greedy Heuristic for KP
def GreedyHeuristicKnapsack(n,W,v,w):
    ratio = v/w
    full_overview = np.array([ratio, v, w])
    sorted_ratio = np.argsort(ratio)[::-1]
    sorted_overview = full_overview[:, sorted_ratio]
    weight_knapsack = 0
    value_knapsack = 0
    i = 0
    greedy_ind = []
    while weight_knapsack < W and i<n:
        #add if item will not exceed the capacity W
        if weight_knapsack+sorted_overview[2,i]<W:
            weight_knapsack+=sorted_overview[2,i] 
            value_knapsack+=sorted_overview[1,i]
            greedy_ind.append(i)
            i+=1
            continue;  
        #keep looping (search remaining items)
        i+=1
    return greedy_ind   
n = 10
w,v,W = GenerateKnapsackData(10, 100, 0.4)

