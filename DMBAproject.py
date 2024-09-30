#Implement correct packages
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gurobipy as gp
import random
import time
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
            i+=1
            weight_knapsack+=sorted_overview[1,i] 
            value_knapsack+=sorted_overview[2,i]
            continue;  
        #keep looping (search remaining items)
        i+=1
    #print(sorted_overview)
    return value_knapsack    

start_time_greedy = time.time()
opt_val_greedy = GreedyHeuristicKnapsack(n,w,v,W)
end_time_greedy = time.time()
run_time_greedy = end_time_greedy - start_time_greedy
print('Optimal Value Greedy Heuristic: {}, in {} seconds.'.format(opt_val_greedy, run_time_greedy))

#%%4. 


'''
#%%Train met ChatGPT

#Define the RNN model
def CreateRecurrentNeuralNetwork(n_input, n_hidden, n_output):
    input_rnn = tf.keras.Input(shape=(None, n_input))  # None for sequence length
    hidden_rnn = tf.keras.layers.GRU(n_hidden, return_sequences=False, return_state=False)(input_rnn)
    output_rnn = tf.keras.layers.Dense(n_output)(hidden_rnn)  # Output layer
    rnn_model = tf.keras.Model(inputs=input_rnn, outputs=output_rnn)
    return rnn_model

# Hyperparameters
n_input = 2  # Current index and remaining capacity
n_hidden = 100  # Number of hidden units
n_output = 2  # Output size: action space (0: do not take item, 1: take item)
eta = 0.001  # Learning rate
gamma = 0.99  # Discount factor

# Initialize RNN model
rnn = CreateRecurrentNeuralNetwork(n_input, n_hidden, n_output)
opt = tf.keras.optimizers.Adam(learning_rate=eta)
loss_fn = tf.keras.losses.MeanSquaredError()

# Memory for the agent
memory = []
max_cells_memory = 10000

# Function to add experiences to memory
def add_memory(s, a, r, next_s, done):
    if len(memory) >= max_cells_memory:
        memory.pop(0)
    memory.append((s, a, r, next_s, done))

# Epsilon-greedy policy for exploration
def epsilon_greedy(s, epsilon):
    # Reshape state to have shape (batch_size, sequence_length, n_input)
    s_seq = np.expand_dims(np.expand_dims(s, axis=0), axis=1)  # Shape becomes (1, 1, 2)
    
    if random.random() > epsilon:
        q_values = rnn(s_seq)  # Shape is (1, n_output)
        action = np.argmax(q_values.numpy())  # Choose action based on Q-values
    else:
        action = np.random.randint(0, n_output)  # Explore: random action
    return action

# Training parameters
epsilon_init = 1.0
min_epsilon = 0.1
fade_eps = 0.995
k = 100  # Batch size for experience replay
n_index = 100  # Number of items (adjustable as needed)
iterations = 1000  # Number of training iterations

#%% Training loop
for iteration in range(iterations):
    index, u = 0, W  # Start with full capacity
    final = False
    cum_r = 0
    epsilon = max(min_epsilon, epsilon_init * (fade_eps ** iteration))  # Epsilon decay

    while not final and index < n_index:
        current_s = np.array([index, u])  # Current state: [item index, remaining capacity]
        a = epsilon_greedy(current_s, epsilon)  # Get action

        # Simulate taking or skipping the item based on action
        if a == 1 and u >= w[index]:
            r = v[index]
            cum_r += r
            next_s = (index + 1, u - w[index])
        else:
            r = 0
            next_s = (index + 1, u)

        done = next_s[0] == n_index
        add_memory(current_s, a, r, next_s, done)  # Store transition in memory

        # Train the RNN using experience replay
        if len(memory) > k:
            memory_retrieval = random.sample(memory, k)
            for sample in memory_retrieval:
                s, a, r, next_s, done = sample

                # Prepare state sequences
                s_seq = np.expand_dims(np.expand_dims(s, axis=0), axis=1)  # Shape (1, 1, 2)
                next_s_seq = np.expand_dims(np.expand_dims(next_s, axis=0), axis=1)  # Shape (1, 1, 2)

                # Q-learning target
                if done:
                    target = r  # No future reward
                else:
                    next_q_values = rnn(next_s_seq).numpy()  # Next Q-values
                    target = r + gamma * np.max(next_q_values)  # Target with discounted future rewards

                # Train the RNN
                with tf.GradientTape() as tape:
                    q_values = rnn(s_seq)  # Current Q-values
                    loss_value = loss_fn(tf.convert_to_tensor([[target]]), q_values[0][a])  # Compute loss
                
                # Apply gradients
                grads = tape.gradient(loss_value, rnn.trainable_variables)
                opt.apply_gradients(zip(grads, rnn.trainable_variables))

        # Move to next state
        index, u = next_s
        print(f"Step: {index}, Remaining Capacity: {u}, Total Reward: {cum_r}")

    print(f"In iteration {iteration + 1}, the total reward is {cum_r}")

#%% Testing function for the trained Q-learning agent
def TestRNN(n_items, W, w, v):
    index, u = 0, W
    cum_value = 0

    while index < n_items:
        current_s = np.array([index, u])  # Current state
        s_seq = np.expand_dims(np.expand_dims(current_s, axis=0), axis=1)  # Shape (1, 1, 2)
        action = np.argmax(rnn(s_seq).numpy())  # Choose best action

        if action == 1 and u >= w[index]:
            cum_value += v[index]
            u -= w[index]

        index += 1

    return cum_value

# Testing the trained Q-learning agent
maximum = TestRNN(n_index, W, w, v)
print("Maximum value obtained:", maximum)
'''

'''
Initial code (doesn't work with the tensors')
#%% 4. Using a NN to approximately solve the Dynamic Programming (Q-learning) - either 4 or 5
#5. USING RL to improve greedy heuristic. - either 4 or 5
#Choose either 4. or 5. not both

#Set n=10000?
#Recurrent Neural Network (idea from https://arxiv.org/pdf/1611.09940, adapted to simpler form with Q-tables)
#testing code (probably not good)
#Not Dense in added hidden layer but GRU for recurrent neural network - make 'note sheet' for agent later (the memory cells)

#Dit werkt nog niet helemaal maar heb er vertrouwen in 
def CreateRecurrentNeuralNetwork(n_input, n_hidden, n_output):
    input_rnn = tf.keras.Input(shape = (None, n_input)) #due to Recurrent layer None, n_input
    hidden_rnn = tf.keras.layers.GRU(n_hidden, return_sequences=True, return_state=True)
    throughput_rnn, state_rnn = hidden_rnn(input_rnn)
    output_rnn = tf.keras.layers.Dense(n_output) #output of 'normal' NN without memory
    q_table = output_rnn(throughput_rnn)
    rnn = tf.keras.Model(inputs=input_rnn, outputs=[q_table, state_rnn])
    return rnn

#Input is current state: i.e. current index and remaining weight left
n_input = 2 
n_hidden = 100 #not W? 
n_output = 2 #0 or 1 of item
eta = 0.001
rnn = CreateRecurrentNeuralNetwork(n_input, n_hidden, n_output)
opt = tf.keras.optimizers.Adam(learning_rate=eta)
loss = tf.keras.losses.MeanSquaredError()

#Exploitation (notebook in which agent keeps track what his best solutions were)
memory = []
max_cells_memory = 10000
#Add trips to memory
def add_memory(s, a, r, next_s, done):
    if len(memory)>= max_cells_memory:
        memory.pop(0) #remove last (minst goede verwijderen lijkt me beter)
    memory.append((s,a,r,next_s, done))
    
#Exploration (pick (1-epsilon)*100% amount of time good solution, otherwise bad solution to explore)
#We use epsilon-greedy policy (see TU Delft article https://pure.tudelft.nl/ws/portalfiles/portal/148757730/978_3_030_86286_2_1.pdf)
def epsilon_greedy(rnn, s, s_hidden, epsilon):
    if random.random()>epsilon: #take random number between 0 and 1
        q_table, _ = rnn(s, s_hidden)
        action = np.argmax(q_table.numpy())
    else:
        action = np.random.randint(0,2) #exclusive high
    return action

gamma = 0.99 
epsilon_init = 1
min_epsilon = 0.1
fade_eps = 0.995
k = 64
iterations = 1000

for iteration in range(iterations):
    #let u remaining capacity
    index, u = 0,W
    final = False
    cum_r = 0
    s_hidden = None #This is memory within the iteration
    s_seq = [] #keep track of all states in the sequence
    while not final and index<n:
        current_s = [index, u]
        s_seq.append(current_s)
        #convert states to useful 
        s_seq_tensor = tf.convert_to_tensor([s_seq], dtype=tf.float32) #dtype toevoegen?
        epsilon = max(min_epsilon, epsilon_init * (fade_eps ** iteration))
        a = epsilon_greedy(rnn, s_seq_tensor, s_hidden, epsilon)
        if a==1 and u >= w[index]:
            r = v[index]
            cum_r += v[index]
            next_s = (index+1, u-w[index])
        else:
            r = 0
            next_s = (index+1, u)
        done = next_s[0] == n #check if next index is n
        next_s_seq = s_seq + [[next_s[0], next_s[1]]]
        next_s_tensor = tf.convert_to_tensor(next_s_seq, dtype=tf.float32) #dtype toevoegen?
        add_memory(s_seq, a, r, next_s_seq, final)
        
        #take random sample from memory
        if len(memory) > k:
            memory_retrieval = random.sample(memory, k)
            #zip is a function to conviently retrieve data reordered
            sample_s_seq, sample_a, sample_r, sample_next_s_seq, sample_done = zip(*memory_retrieval) 
            sample_s_seq_tensor = tf.convert_to_tensor(sample_s_seq, dtype=tf.float32) #, dtype=tf.float32 toevoegen?
            sample_a_tensor = tf.convert_to_tensor(sample_a, dtype=tf.float32) #dtype=tf.float toevoegen?
            sample_r_tensor = tf.convert_to_tensor(sample_r, dtype=tf.float32) #dtype=tf.float toevoegen?
            sample_next_s_seq_tensor = tf.convert_to_tensor(sample_next_s_seq, dtype=tf.float32) #dtype=tf.float toevoegen?
            sample_done_tensor = tf.convert_to_tensor(sample_done, dtype=tf.float32) #dtype=tf.float toevoegen?
            with tf.GradientTape() as gradient_calc: #weet niet precies wat dit doet later uitzoeken, iets met alles in het blok in een keer gradients uitrekenen
                #retrieve current q_f
                q_table, _ = rnn(sample_s_seq_tensor)
                #categorize variables with one_hot
                sample_a_categorization = tf.one_hot(sample_a_tensor, 2)
                #update q-values based on action
                q_values_update = tf.reduce_sum(q_table * sample_a_categorization, axis=-1)
                #take max q-value to use in the next state (improve <-- RL)
                next_q_table, _ = rnn(sample_next_s_seq_tensor)
                selected_q_value = tf.reduce_max(next_q_table, axis=-1)
                #set desired Q-value
                desired_Q = sample_r_tensor + gamma*selected_q_value * (1-sample_done_tensor)
                L = loss(desired_Q, q_values_update)
            gradients = gradient_calc.gradient(L, rnn.trainable_variables)
            opt.apply_gradients(zip(gradients, rnn.trainable_variables))
        #move to next state
        s = next_s
        i,u = next_s_tensor
    print("In iteration {}, the total reward is {}".format(iteration+1, cum_r))
    
def TestRNN(q_network, n_items, W, w, v):
    index, u = 0, W
    cum_value = 0
    hidden_s = None
    s_seq = []
    while index<n_items:
        s_seq.append([index, u])
        s_seq_tensor = tf.convert_to_tensor([s_seq], dtype=tf.float32)
        #use Greedy approach first (I really don't know whether this is more 4 or 5)
        q_table, hidden_s = q_network(s_seq_tensor, hidden_s)
        a = np.argmax(q_table.numpy())
        if a == 1 and u >= w[index]:
            cum_value += v[index]
            u -= w[index]
        index+=1
    return cum_value
maximum = TestRNN(rnn, n, W, w, v)
print(maximum)
'''