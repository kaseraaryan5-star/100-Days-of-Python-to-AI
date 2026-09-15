import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# 1. Generate data
X, y = make_classification(
    n_samples=100, 
    n_features=2, 
    n_informative=2, 
    n_redundant=0, 
    n_classes=2, 
    n_clusters_per_class=1, 
    random_state=41, 
    hypercube=False, 
    class_sep=15
)

# 2. Plotting (Optional visual check)
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter', s=100)
print(plt.show())
# Note: plt.show() blocks code execution, so it is commented out 
# while training. You can uncomment it to see the initial plot.
# plt.show() 

# 3. Perceptron Training Function
def perceptron(X, y):
    # Insert step 1 at index 0 to turn X into a 3D-like array format if needed,
    # or simply iterate through the NumPy array coordinates properly.
    X = np.insert(X, 0, 1, axis=1)
    weights = np.ones(X.shape[1])
    lr = 0.1
    
    for epoch in range(1000):
        for i in range(len(X)):
            # Dot product of weights and features
            z = np.dot(weights, X[i])
            
            # Activation condition (Heaviside step function)
            if z * y[i] < 0:
                weights = weights + lr * y[i] * X[i]
                
    return weights[1:], weights[0]

# 4. Run the model
w1_w2, b = perceptron(X, y)
print("Weights:", w1_w2)
print("Bias:", b)

x_input = np.linspace(-3,3,100)
y_input = w1_w2[0] * x_input + b

plt.figure(figsize=(10,6))
plt.plot(x_input, y_input,color='red',linewidth=3)
plt.scatter(X[:,0],X[:,1],c=y,cmap='winter',s=100)
plt.ylim(-3,2)
print(plt.show())