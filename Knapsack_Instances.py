import numpy as np
import math


def circle(n,R,d = 2/3):
    # Paper chooses d = 2/3
    weights = np.random.randint(1,R,n)
    profits = d * np.sqrt(4*R**2 - (weights - 2*R)**2)
    return weights,profits
'''
def profit_ceil(n,R,d = 3):
    # Paper chooses d = 3
    weights = np.random.randint(1,R,n)
    profits = d*np.ceil(weights/d)
    return weights,profits
'''
def correlated_instances(n,w_max):
    weights = np.random.randint(1,w_max,n)
    profits = weights + np.ceil(w_max/10)
    return weights,profits
    
def spanner(n,R,v,m = 10):
    weights_span = np.random.randint(1,R,v)
    profits_span = np.zeros(v)
    for i in range(v):
        #weakly correlated
        profits_span[i] = np.random.randint(max(1,weights_span[i] - R/10), weights_span[i]+R/10)
    norm_weights_span = np.ceil(2*weights_span/m)
    norm_profits_span = np.ceil(2*profits_span/m)
    weights = np.copy(norm_weights_span)
    profits = np.copy(norm_profits_span)
    while np.shape(weights)[0] < n:
        index = np.random.randint(1,v)
        alpha = np.random.randint(1,m)
        profit_new = alpha * profits[index]
        weight_new = alpha * weights[index]
        weights = np.append(weights,weight_new)
        profits = np.append(profits,profit_new)
    return weights,profits

print(profit_ceil(100,100))

w,v = profit_ceil(100,100)
    
        
        
    

