#Implement correct packages
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gurobipy as gp
np.random.seed(6)

#%% 6. Generate data
def GenerateKnapsackData(n, w_max, alpha):
    w = np.zeros(n)
    v = np.zeros(n)
    for i in range(n):
        w[i] = np.random.randint(low=1, high=w_max) #Make sure we do not have 0's. Divsision by 0 error
        v[i] = w[i] + np.random.randint(low=0, high=w_max//5)  #we let v[i] be correlated with w[i] with added randomness. 
    w_sum = np.sum(w)
    W = int(round(alpha * w_sum,0))
    return w,v,W
    
n=100
w_max = 100
alpha = 0.3
w,v,W = GenerateKnapsackData(n, w_max, alpha)

#%% 1. Binary Programming of KP
def BinaryProgrammingKnapsack(n,w,v,W):
    knapsack = gp.Model("Knapsack Binary Programming")
    
    # Indices
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
    
    for i in index:
        opt = x[i].X
        #print("x_{}: {}".format(i,opt))
    opt_val = knapsack.ObjVal
    print("Optimal Value Gurobi: {}".format(opt_val))
    return opt_val

opt_val_bin = BinaryProgrammingKnapsack(n, w, v, W)

#%% 2. Dynamic Programming of KP
def DynamicProgrammingKnapsack(n,w,v,W):
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
    print('Optimal Value Dynamic Programming: {}'.format(opt))
    return opt

opt_val_dyn = DynamicProgrammingKnapsack(n,w,v,W)
        
#%% 3. Greedy Heuristic for KP
def GreedyHeuristicKnapsack(n,w,v,W):
    ratio = v/w
    full_overview = np.array([ratio, v, w])
    sorted_ratio = np.argsort(ratio)[::-1]
    sorted_overview = full_overview[:, sorted_ratio]
    weight_knapsack = 0
    value_knapsack = 0
    i = 0
    while weight_knapsack < W and i<n:
        #add if item will not exceed the capacity W
        if weight_knapsack+sorted_overview[1,i]<W:
            i+=1
            weight_knapsack+=sorted_overview[1,i] 
            value_knapsack+=sorted_overview[2,i]
            continue;  
        #keep looping (search remaining items)
        i+=1
    #print(sorted_overview)
    print(value_knapsack)
    return value_knapsack    
    
opt_val_greedy = GreedyHeuristicKnapsack(n,w,v,W)

#%% 4. Using a NN to approximately solve the Dynamic Programming (Q-learning)


#No 5. (Using RL to improve the heuristic with Greedy Heuristic)


