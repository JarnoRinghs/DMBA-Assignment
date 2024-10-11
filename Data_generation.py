import numpy as np

#%% 6. Generate data
def GenerateKnapsackData(n,w_max,alpha):
    weights = np.random.randint(1,w_max,n)
    profits = weights + np.ceil(w_max/10)
    W = int(round(alpha*np.sum(weights),0))
    return weights,profits,W
alpha = 0.4
