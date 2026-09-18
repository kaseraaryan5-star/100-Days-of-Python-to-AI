import numpy as np
import pandas as pd

df=pd.DataFrame([[8,8,4],[7,9,5],[6,10,6],[5,12,7]],columns=['cgpa','profile_score','lpa'])
print(df)

def initialize_parameters(layer_dims):
  np.random.seed(3)
  parameters={}
  L=len(layer_dims)

  for l in range(1,L):
    parameters['W'+str(l)]=np.ones((layer_dims[l-1],layer_dims[l]))*0.1
    parameters['b'+str(l)]=np.zeros((layer_dims[l],1))

  return parameters

print(initialize_parameters([2,2,1]))

def linear_forward(A_prev,W,b):
  Z=np.dot(W.T,A_prev)+b
  return Z

# Forward Prop
def L_layer_forward(X, parameters):

  A = X
  L = len(parameters) // 2                  # number of layers in the neural network

  for l in range(1, L+1):
    A_prev = A
    Wl = parameters['W' + str(l)]
    bl = parameters['b' + str(l)]
    #print("A"+str(l-1)+": ", A_prev)
    #print("W"+str(l)+": ", Wl)
    #print("b"+str(l)+": ", bl)
    #print("--"*20)

    A = linear_forward(A_prev, Wl, bl)
    #print("A"+str(l)+": ", A)
    #print("**"*20)

  return A,A_prev

X = df[['cgpa', 'profile_score']].values[0].reshape(2,1) # Shape(no of features, no. of training example)
y = df[['lpa']].values[0][0]

# Parameter initialization
parameters = initialize_parameters([2,2,1])

y_hat,A1 = L_layer_forward(X, parameters)

y_hat=y_hat[0][0]
print(A1)

def update_parameters(parameters,y,y_hat,A1,X):
  parameters['W2'][0][0] = parameters['W2'][0][0] + (0.001 * 2 * (y - y_hat)*A1[0][0])
  parameters['W2'][1][0] = parameters['W2'][1][0] + (0.001 * 2 * (y - y_hat)*A1[1][0])
  parameters['b2'][0][0] = parameters['W2'][1][0] + (0.001 * 2 * (y - y_hat))

  parameters['W1'][0][0] = parameters['W1'][0][0] + (0.001 * 2 * (y - y_hat)*parameters['W2'][0][0]*X[0][0])
  parameters['W1'][0][1] = parameters['W1'][0][1] + (0.001 * 2 * (y - y_hat)*parameters['W2'][0][0]*X[1][0])
  parameters['b1'][0][0] = parameters['b1'][0][0] + (0.001 * 2 * (y - y_hat)*parameters['W2'][0][0])

  parameters['W1'][1][0] = parameters['W1'][1][0] + (0.001 * 2 * (y - y_hat)*parameters['W2'][1][0]*X[0][0])
  parameters['W1'][1][1] = parameters['W1'][1][1] + (0.001 * 2 * (y - y_hat)*parameters['W2'][1][0]*X[1][0])
  parameters['b1'][1][0] = parameters['b1'][1][0] + (0.001 * 2 * (y - y_hat)*parameters['W2'][1][0])

update_parameters(parameters,y,y_hat,A1,X)

print(parameters)

X=df[['cgpa','profile_score']].values[0].reshape(2,1)
y=df[['lpa']].values[0][0]

parameters=initialize_parameters([2,2,1])

y_hat,Al=L_layer_forward(X,parameters)
y_hat=y_hat[0][0]

update_parameters(parameters,y,y_hat,Al,X)

print(parameters)

X=df[['cgpa','profile_score']].values[1].reshape(2,1)
y=df[['lpa']].values[1][0]


y_hat,Al=L_layer_forward(X,parameters)
y_hat=y_hat[0][0]

update_parameters(parameters,y,y_hat,Al,X)

print(parameters)

X=df[['cgpa','profile_score']].values[2].reshape(2,1)
y=df[['lpa']].values[2][0]


y_hat,Al=L_layer_forward(X,parameters)
y_hat=y_hat[0][0]

update_parameters(parameters,y,y_hat,Al,X)

print(parameters)

X=df[['cgpa','profile_score']].values[3].reshape(2,1)
y=df[['lpa']].values[3][0]


y_hat,Al=L_layer_forward(X,parameters)
y_hat=y_hat[0][0]

update_parameters(parameters,y,y_hat,Al,X)

print(parameters)

# epoch implementation
parameters=initialize_parameters([2,2,1])

epochs=5

for i in range(epochs):
  Loss=[]

  for j in range(df.shape[0]):

    X=df[['cgpa', 'profile_score']].values[j].reshape(2,1)
    y=df[['lpa']].values[j][0]

    #parameter initialization

    y_hat,Al=L_layer_forward(X,parameters)
    y_hat=y_hat[0][0]

    update_parameters(parameters,y,y_hat,Al,X)

    Loss.append((y-y_hat)**2)

  print('Epoch - ',i+1,'Loss - ',np.array(Loss).mean())

print(parameters)

