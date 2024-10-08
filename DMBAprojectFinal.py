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
#np.random.seed(6)

#%% 6. Generate data
'''
def GenerateKnapsackData(n, w_max, alpha):
    w = np.zeros(n)
    v = np.zeros(n)
    for i in range(n):
        w[i] = np.random.randint(low=1, high=w_max) #Make sure we do not have 0's. Divsision by 0 error
        v[i] = w[i] + np.random.randint(low=0, high=w_max//5)  #we let v[i] be correlated with w[i] with added randomness. 
    w_sum = np.sum(w)
    W = int(round(alpha * w_sum,0))
    return w,v,W
'''
def GenerateKnapsackData(n,w_max,alpha):
    weights = np.random.randint(1,w_max,n)
    profits = weights + np.ceil(w_max/10)
    W = int(round(alpha*np.sum(weights),0))
    return weights,profits,W
    
n=10
w_max =100
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
    return opt_val

start_time_bin = time.time()
opt_val_bin = BinaryProgrammingKnapsack(n, w, v, W)
end_time_bin = time.time()
run_time_bin = end_time_bin - start_time_bin
print("Optimal Value Gurobi (Binary Programming): {}, in {} seconds.".format(opt_val_bin, run_time_bin))

#%% 2. Dynamic Programming of KP
#If we set n=10000, we get memory error after a while
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
        #print('Point: {}.'.format(i))
    opt = M[(n,W)]
    return opt

start_time_dyn = time.time()
opt_val_dyn = DynamicProgrammingKnapsack(n,w,v,W)
end_time_dyn = time.time()
run_time_dyn = end_time_dyn - start_time_dyn
print('Optimal Value Dynamic Programming: {}, in {} seconds.'.format(opt_val_dyn, run_time_dyn))
        
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
            weight_knapsack+=sorted_overview[1,i] 
            value_knapsack+=sorted_overview[2,i]
            i+=1
            continue;  
        #keep looping (search remaining items)
        i+=1
    return value_knapsack    

start_time_greedy = time.time()
opt_val_greedy = GreedyHeuristicKnapsack(n,w,v,W)
end_time_greedy = time.time()
run_time_greedy = end_time_greedy - start_time_greedy
print('Optimal Value Greedy Heuristic: {}, in {} seconds.'.format(opt_val_greedy, run_time_greedy))
#%%4. Using a NN to approximately solve the Dynamic Programming (NDP) - We will do exercise 4 not 5

def NeuralDynamicProgrammingKnapsack(n,w,v,W, iterations):
    ##Initlaliseer parameters voor q-learning
    epsilon = 0.9
    beta = 0.995
    gamma = 0.6
    num_episodes = 25
'''  
    # Initialize the model
    model = keras.Sequential()
    input_shape = 4
    model.add(layers.Input(shape=(input_shape,)))
    model.add(layers.Dense(20, activation="relu"))
    model.add(layers.Dense(2, activation="linear"))'Wrm die 10 weer?'
    model.compile(loss="mse", optimizer="adam", metrics=['mse'])
'''
    best_items = []
    best_value = 0
    
    for j in range(num_episodes):
        print('Iteration {} is running.'.format(j+1))
        epsilon = max(epsilon*beta,0.1)
        items = []
        w_left = W
        cum_value = 0
        for i in range(n):
            state = np.array([w[i], v[i], v[i] / w[i], w_left])
            #epsilon-greedy
            if np.random.random() < epsilon:
                action = np.random.randint(0, 2)
            else:
                action = np.argmax(model.predict(state.reshape(1, -1), verbose=0))
    
            if action == 1:
                w_left -= w[i]
                if w_left < 0:
                    reward = -10 *v[i]
                else:
                    reward = v[i]
                    cum_value += reward
                    items.append(i)
            else:
                reward = 0
            
            if i < n-1:
                next_state = np.array([w[i+1], v[i+1], v[i+1]/w[i+1], w_left])
                target_value = reward + gamma * np.max(model.predict(next_state.reshape(1, -1), verbose=0))
            else:
                target_value = reward
            q_values = model.predict(state.reshape(1,-1), verbose=0)
            q_values[0][action] = target_value
            model.fit(state.reshape(1, -1), q_values, epochs=1, verbose=0)
        if cum_value > best_value:
            best_value = cum_value
            best_items = items[:]
    end_value = sum([v[i] for i in best_items])
    return end_value

iterations = 25
start_time_qlearning = time.time()
opt_val_qlearning = NeuralDynamicProgrammingKnapsack(n,w,v,W, iterations)
end_time_qlearning = time.time()
run_time_qlearning = end_time_qlearning - start_time_qlearning
print("For n = {} and W = {}:".format(n,W))
print("Optimal Value Gurobi (Binary Programming): {}, in {} seconds.".format(opt_val_bin, run_time_bin))
print('Optimal Value Dynamic Programming: {}, in {} seconds.'.format(opt_val_dyn, run_time_dyn))
print('Optimal Value Greedy Heuristic: {}, in {} seconds.'.format(opt_val_greedy, run_time_greedy))
print('Optimal Value Q-Learning: {}, in {} seconds.'.format(opt_val_qlearning, run_time_qlearning))
