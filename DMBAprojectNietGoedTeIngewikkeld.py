#Implement correct packages
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gurobipy as gp
import random
import time
import math
import os
np.random.seed(6)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
    
n=20
w_max = 100
alpha = 0.3
# w,v,W = GenerateKnapsackData(n, w_max, alpha)

from Knapsack_Instances import profit_ceil
w,v = profit_ceil(n,1000)
W = math.floor(alpha * np.sum(w))


print(w)
print(v)
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
#Training
#Policy Network
def policy_network(n_hidden, n_input):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(n_hidden, activation='sigmoid', input_shape=(n_input,)))
    model.add(tf.keras.layers.Dense(2, activation='softmax')) #output actions
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def Q_network(n_hidden, n_input):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(n_hidden, activation='sigmoid', input_shape = (n_input,)))
    model.add(tf.keras.layers.Dense(2, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    return model

def reward(s, a, item, W, w, v):
    if a == 1:
        if w[item] <= s[item,1]:
            r = v[item]
            s[item,0] = 1
            s[item,1] = s[item,1] - w[item]
        else:
            r = 0 #Penalty
    else:
        r=0
    return s,r
        
def train_knapsack(episodes, W, n_items, w, v):
    iterations = 3
    memory = []
    gamma = 0.9
    beta = 0.95
    epsilon = 0.95
    n_hidden = 16
    n_policy = n_items * 2 
    n_q = n_items*2 #might be n_items*4+1
    policy_NN = policy_network(n_hidden, n_policy)
    Q_NN = Q_network(n_hidden, n_q)
    for episode in range(episodes):
        s= np.zeros((n_items, 2))
        s[0,1] = W
        cum_r = 0
        for t in range(n_items):
            s_t = s.reshape(1,-1)
            a_probabilities = policy_NN.predict(s_t, verbose=0)
            if epsilon > np.random.rand(): #epsilon greedy policy (take some option with probability epsilon) - ADAPT STILL 
                a = np.argmax(a_probabilities[0])
            else:
                a = np.random.choice(2) #to max?
            next_s, r = reward(s.copy(), a, t, W, w, v)
            cum_r += r
            s_t_next = next_s.reshape(1, -1)
            r_t_next = np.array([r])
            memory.append((s_t, a, s_t_next, r_t_next, r))
            #Update neural network for Q-values with previous experiences
            if len(memory) > 1:
                k = min(len(memory), 16)
                batch = random.sample(memory, k) #get minibatch
                for prev_s_t, prev_a, prev_s_t_next, prev_r_t_next, prev_r in batch:
                    prev_q_values = Q_NN.predict(prev_s_t, verbose=0)
                    q_values_next = Q_NN.predict(prev_s_t_next, verbose=0)
                    q_value_goal = prev_r + gamma * np.max(q_values_next[0])
                    prev_q_values[0][prev_a] = q_value_goal
                    Q_NN.fit(prev_s_t, prev_q_values, epochs=iterations, verbose=0)
                    
            #update neural network for Q-values with the current experiences
            q_values_current = Q_NN.predict(s_t, verbose=0)
            q_values_next = Q_NN.predict(s_t_next, verbose = 0)
            if t == n_items -1: #stop q-value updating in last item (no later q-values)
                q_value_goal_current = r
            else:
                q_value_goal_current = r + gamma * np.max(q_values_next[0])
            q_values_current[0][a] = q_value_goal_current
            Q_NN.fit(s_t, q_values_current, epochs=iterations, verbose=0)
            
            #Update policy network
            action_converted = np.zeros((1,2))
            action_converted[0,a] = 1
            policy_NN.fit(s_t, action_converted, epochs=iterations, verbose=0)
            s = next_s
        epsilon = max(0.1, epsilon*beta) #decay the epsilon
        print('In Episode {}, we have cumulative reward {}.'.format(episode+1,cum_r))
    return policy_NN, Q_NN

episodes = 10
start_time_training = time.time()
opt_policy_NN, opt_Q_NN = train_knapsack(episodes, W, n, w, v)
end_time_training = time.time()
training_time = end_time_training - start_time_training
print('It took {} minutes to train the reinforcement learning.'.format(round(training_time/60,1)))

#%%
#Implementing
#Calculate optimal value using the trained neural networks here
def NeuralDynamicProgramming(n_items, w, v, W, policy_NN, Q_NN):
    s = np.zeros((n_items,2))
    s[0,1] = W
    opt_val = 0
    for t in range(n_items):
        s_t = s.reshape(1,-1)
        a_probabilities = policy_NN.predict(s_t, verbose=0)
        a= np.argmax(a_probabilities[0])
        next_s, r = reward(s.copy(), a, t,W, w, v)
        opt_val += r
        s = next_s
        q_values = Q_NN.predict(s_t, verbose=0).flatten()
        q_value = q_values[a]
        print(f"State: {s_t.flatten()}, Action: {a}, Q-value: {q_value}")
    return opt_val

start_time_nn = time.time()
opt_val_nn = NeuralDynamicProgramming(n, w, v, W, opt_policy_NN, opt_Q_NN)
end_time_nn = time.time()
run_time_nn = end_time_nn - start_time_nn
print("Optimal Value Gurobi (Binary Programming): {}, in {} seconds.".format(opt_val_bin, run_time_bin))
print('Optimal Value Dynamic Programming: {}, in {} seconds.'.format(opt_val_dyn, run_time_dyn))
print('Optimal Value Greedy Heuristic: {}, in {} seconds.'.format(opt_val_greedy, run_time_greedy))
print("Optimal Value Approximative Neural Network - Q-learning approach: {}, in {} seconds.".format(opt_val_nn, run_time_nn))





