import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

or_data = pd.DataFrame()
and_data = pd.DataFrame()
xor_data = pd.DataFrame()

or_data['input1']=[1,1,0,0]
or_data['input2']=[1,0,1,0]
or_data['output']=[1,1,1,0]

and_data['input1']=[1,1,0,0]
and_data['input2']=[1,0,1,0]
and_data['output']=[1,0,0,0]

xor_data['input1']=[1,1,0,0]
xor_data['input2']=[1,0,1,0]
xor_data['output']=[0,1,1,0]

print(or_data)
print(sns.scatterplot(x=or_data['input1'],y=or_data['input2'],hue=or_data['output'],s=200))
print(plt.show())

print(and_data)
print(sns.scatterplot(x=and_data['input1'],y=and_data['input2'],hue=and_data['output'],s=200))
print(plt.show())

print(xor_data)
print(sns.scatterplot(x=xor_data['input1'],y=xor_data['input2'],hue=xor_data['output'],s=200))
print(plt.show())

from sklearn.linear_model import Perceptron
clf1 = Perceptron()
clf2 = Perceptron()
clf3 = Perceptron()

clf1.fit(or_data.iloc[:,0:2].values,or_data.iloc[:,-1].values)
clf1.fit(and_data.iloc[:,0:2].values,and_data.iloc[:,-1].values)
clf1.fit(xor_data.iloc[:,0:2].values,xor_data.iloc[:,-1].values)

print(clf1.coef_)
print(clf1.intercept_)
x=np.linspace(-1,1,5)
y=-x+1

print(plt.plot(x,y))
print(sns.scatterplot(x=or_data['input1'],y=or_data['input2'],hue=or_data['output'],s=200))
print(plt.show())

print(clf1.coef_)
print(clf1.intercept_)
x1=np.linspace(-1,1,5)
y1=-x+0.5

print(plt.plot(x1,y1))
print(sns.scatterplot(x=or_data['input1'],y=or_data['input2'],hue=or_data['output'],s=200))
print(plt.show())

print(clf3.coef_)
print(clf3.intercept_)
print(plot_decisions_region(xor_data.iloc[:,0:2].values,xor_data.iloc[:,-1].values,clf=clf3,legend=2))
print(plt.show())