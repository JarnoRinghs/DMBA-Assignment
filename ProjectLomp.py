from functionsLomp import GenerateKnapsackData
import numpy as np
import math
from functionsLomp import DynamicProgrammingKnapsack
from tensorflow import keras
from tensorflow.keras import layers


n = 10
w_max = 100
alpha = 0.1
W = math.ceil(w_max*n * alpha)
size_training = 1000
size_test = 100


def Generate_dataset(size_data,n,w_max,W):
    x_matrix = np.zeros((size_data,n,3))
    y = np.zeros(((n+1)*(W+1),size_data))
    for i in range(size_data):
        w,v = GenerateKnapsackData(n,w_max)
        x_matrix[i,:,0] = w
        x_matrix[i,:,1] = v
        x_matrix[i,:,2] = v/w
        
        y_i_dict = DynamicProgrammingKnapsack(n,w,v,W,1)
        y_i = np.array([y_i_dict[n_,w_] for w_ in range(W+1) for n_ in range(n+1)])
        y[:,i] = y_i
        print(i)

    sorted_x_matrix = np.zeros_like(x_matrix)  # Create a new array to store sorted data
    for i in range(size_data):
        sorted_indices = np.argsort(x_matrix[i, :, 2])  # Get the indices that would sort the third column
        sorted_x_matrix[i] = x_matrix[i, sorted_indices]  # Sort the 2D array using the indices
        
        
    x = np.array([sorted_x_matrix[i].ravel() for i in range(size_data)]) # or use .flatten()
    
    return x, np.transpose(y)

x_train,y_train = Generate_dataset(size_training,n,w_max,W)
print(np.shape(x_train))
print(np.shape(y_train))
x_test,y_test = Generate_dataset(size_test,n,w_max,W)

nrlayers = 2
def Lomp_model(n,W,x_train,y_train,nrlayers,rate):
    # This function builds the model with nrlayes layers and nrnodes nodes in each layer
    # nrlayers is a number and nrnodes is a vector specifying how many nodes each layer should contain
    model = keras.Sequential()
    size_output = (n+1)*(W+1)

    input_shape = n*3
    
    # add the layers
    model.add(layers.Input(shape = (input_shape,)))
    for i in range(nrlayers):
        model.add(layers.Dense(input_shape-10*i,activation = "relu"))
    model.add(layers.Dense(size_output, activation="relu"))

      # train model
    batch_size = 16 # Based on SGD --> 1 gradient is calculated using 32 data points every time
    #so in each epoch, training_size/batch_size many gradient calculations and weight updates
    epochs = 10 # How many times to go through the data to complete the SGD

    opt = keras.optimizers.SGD(learning_rate=rate)
    model.compile(loss="mean_squared_error",
                  optimizer=opt,
                  metrics=["mean_squared_error"])

    # fit the model, note that keras takes care of the train set/validation set
    # use model.fit with a validation_split of 0.1
    model.fit(x_train, y_train, validation_split=0.1, batch_size=batch_size, epochs=epochs)

    
    
    return model

test = Lomp_model(n,W,x_train,y_train,2,0.02)
# Use the trained model to make predictions (reshape the input for correct shape)
test_predict = test.predict(x_test[0,:].reshape(1,-1))

# Reshape the prediction to (n+1, W+1)
reshaped_predict = test_predict

# Print the reshaped prediction and the corresponding actual y_test value
print(reshaped_predict)
MSE = np.sum((reshaped_predict - y_test[0,:])**2)/111
print(MSE)  # Reshape y_test for comparison
