import numpy as np 
import pandas as pd

df = pd.DataFrame([[8,8,4],[7,9,5],[6,10,6],[5,12,7]],columns=['cgpa','profile_score','lpa'])
print(df)

def initialize_parameters(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W'+ str(1)] = np.ones((layer_dims[l-1],layer_dims[1]))*0.2
        parameters['b'+ str(1)] = np.zeros((layer_dims[1],1))

    return parameters

print(initialize_parameters([2,2,1]))