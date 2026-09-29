import numpy as np
import pandas as pd

df = pd.read_csv('/Users/aryankasera/Desktop/100-Days-AI/07_Machine_learning.py/customer.csv')
print(df.sample(5))

df = df.iloc[:,2:]
print(df.head())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df.iloc[:,0:2],df.iloc[:,-1],test_size=0.2)
print(X_train)

from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(categories=[['Poor','Average','Good'],['School','UG','PG']])
oe.fit(X_train)
OrdinalEncoder(categories=[['Poor', 'Average', 'Good'], ['School', 'UG', 'PG']])
X_train = oe.transform(X_train)
print(X_train)

print(oe.categories_)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y_train)
print(le.classes_)
y_train = le.transform(y_train)
y_test = le.transform(y_test)
print(y_train)