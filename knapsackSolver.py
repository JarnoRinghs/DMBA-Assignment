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
#%%4. Using a NN to approximately solve the Dynamic Programming (NDP) - We will do exercise 4

def knapsackSolver(n,W,v,w):
    ##Initialize parameters voor q-learning
    epsilon = 0.9
    beta = 0.995
    gamma = 0.6
    num_episodes = 10
    # Initialize the model
    model = keras.Sequential()
    input_shape = 4
    model.add(layers.Input(shape=(input_shape,)))
    model.add(layers.Dense(20, activation="relu"))
    model.add(layers.Dense(2, activation="linear"))
    model.compile(loss="mse", optimizer="adam", metrics=['mse'])
    best_items = []
    best_value = 0
    
    for j in range(num_episodes):
        print('Iteration {} is running.'.format(j+1))
        epsilon = max(epsilon*beta,0.1)
        items = []
        w_left = W
        cum_value = 0
        index_episode = []
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
                index_episode.append(i)
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
            best_index = index_episode
    end_value = sum([v[i] for i in best_items])
    return best_index