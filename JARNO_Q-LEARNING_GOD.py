import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gurobipy as gp
import random
import time
import math
from tensorflow import keras
from tensorflow.keras import layers


n = 10
w_max = 10000000
alpha = 0.3

def GenerateKnapsackData(n, w_max):
    w = np.zeros(n)
    v = np.zeros(n)
    for i in range(n):
        w[i] = np.random.randint(low=1, high=w_max)  # Make sure we do not have 0's. Division by 0 error
        v[i] = w[i] + np.random.randint(low=0, high=w_max//5)  # v[i] is correlated with w[i] with added randomness. 
    return w, v

w, v = GenerateKnapsackData(n, w_max)
W = math.floor(alpha * np.sum(w))

def DynamicProgrammingKnapsack(n, w, v, W, return_M=False):
    M = {}
    for j in range(W + 1):
        M[(0, j)] = 0
    for i in range(1, n + 1):
        for j in range(W + 1):
            if w[i-1] > j:
                M[(i, j)] = M[(i-1, j)]
            else:
                M[(i, j)] = max(M[(i - 1, j)], v[i-1] + M[(i-1, j - w[i-1])])
    opt = M[(n, W)]
    
    if return_M: 
        return M
    return opt

start_time_dyn = time.time()
opt_val_dyn = DynamicProgrammingKnapsack(n, w, v, W)
end_time_dyn = time.time()
run_time_dyn = end_time_dyn - start_time_dyn
print('Optimal Value Dynamic Programming: {}, in {} seconds.'.format(opt_val_dyn, run_time_dyn))


start_time_qlearning = time.time()
##Initlaliseer parameters voor q-learning
eps = 0.5
eps_decay_factor = 0.95
discount_factor = 0.95
num_episodes = 50

# Initialize the model
model = keras.Sequential()
input_shape = 4
model.add(layers.Input(shape=(input_shape,)))
model.add(layers.Dense(3, activation="relu"))
model.add(layers.Dense(2, activation="linear"))
model.compile(loss="mse", optimizer="adam", metrics=['mse'])

# Het idee in het kort:
# Onze tabel vervangen we door een neural network. Gegeven de value, weight, ratio en weight left in de knapsack,
# doet deze een expected return bepalen voor de actie om hem erin te stoppen of niet. 
# Vervolgens pakken we met een probability van epsilon een random actie en anders de optimale
# Deze actie geven we een reward (value als het past en, -1* w_max als het item niet past)
# Vervolgens is de value die we eigenlijk willen dat uit ons NN komt deze reward
# plus de expected return van de optimale actie in de state waar je in terecht komt (deze berekenen we ook dmv ons NN)
# Vervolgens doen we ons NN op deze target value fitten
# Dit doen we voor ieder object.
# In totaal beginnen we num_episodes keer opnieuw aan het vullen
# Het belangrijke: We koppelen hier onze estimation los van de grootte van W
# num_episodes kunnen we zelf bepalen. HIERDOOR IS DE Q-LEARNING SNELLER VOOR HELEV GROTE WAARDES VAN W

for j in range(num_episodes):
    print(j)
    # Hanteer epsilon greedy policy met decaying epsilon
    eps *= eps_decay_factor
    if j == num_episodes-1:
    # Selecteer items voor solution pas in laatste episode
        items = []
    w_left = W
    for i in range(n):
        state = np.array([w[i], v[i], v[i] / w[i], w_left])
        #epsilon-greedy
        if np.random.random() < eps:
            action = np.random.randint(0, 2)
        else:
            action = np.argmax(model.predict(state.reshape(1, -1)))

        #Reward is value als de item past zo niet punish met 0.1*w_max
        if action == 1:
            w_left -= w[i]
            if w_left < 0:
                reward = -0.1 * w_max
            if j == num_episodes-1:
            #In de laatste episode wordt items onze lijst met items die we selecteren
                items.append(i)
            else:
                reward = v[i]
        else:
            reward = 0
        
        if i < n-1:
        #Bepaal target value. 
        # Laatste item in de rij is laatste state. Dus daar is de target_value alleen de reward
            next_state = np.array([w[i+1], v[i+1], v[i+1]/w[i+1], w_left])
            target_value = reward + discount_factor * np.max(model.predict(next_state.reshape(1, -1)))
        else:
            target_value = reward
        # Fit het model op de target value. Q-LEARNING!!!!!!
        model.fit(state.reshape(1, -1), np.array([target_value]), epochs=1, verbose=0)

opt_val_qlearning = sum([v[i] for i in items])
end_time_qlearning = time.time()
run_time_qlearning = end_time_qlearning - start_time_qlearning
print('Optimal Value Q-Learning: {}, in {} seconds.'.format(opt_val_qlearning, run_time_qlearning))
