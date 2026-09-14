import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/aryankasera/Desktop/100-Days-AI/08_Deep_Learning.py/placement (1).csv')
print(df.head())

print(df.shape)

print(sns.scatterplot(x=df['cgpa'],y=df['resume_score'],hue=df['placed']))
print(plt.show())

X = df.iloc[:,0:2]
Y = df.iloc[:,-1]

from sklearn.linear_model import Perceptron
p = Perceptron()

print(p.fit(X,Y))
print(p.coef_)
print(p.intercept_)

from mlxtend.plotting import plot_decision_regions
print(plot_decision_regions(X.values,Y.values,clf=p,legend=2))
print(plt.show())

