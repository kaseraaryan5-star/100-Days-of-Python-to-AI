import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/aryankasera/Desktop/100-Days-AI/07_Machine_learning.py/train.csv')
print(df.head())

# 1.Categorical Data
# a.Countplot
print(sns.countplot(x='Survived',data=df))
print(plt.show())

print(sns.countplot(x='Pclass',data=df))
print(plt.show())

print(sns.countplot(x='Sex',data=df))
print(plt.show())

# d.PieChart
print((df['Survived']).value_counts().plot(kind='pie',autopct='%.2f'))
print(plt.show())

print((df['Pclass']).value_counts().plot(kind='pie',autopct='%.2f'))
print(plt.show())

print((df['Sex']).value_counts().plot(kind='pie',autopct='%.2f'))
print(plt.show())



# 2.Numerical Data
# a.Histogram
print(plt.hist(df['Age'],bins=5))
print(plt.show())

# b.Displot
print(sns.displot(df['Age']))
print(plt.show())

# c.Boxplot
print(sns.boxplot(df['Fare']))
print(plt.show())

print(sns.boxplot(df['Age']))
print(plt.show())


print(df['Age'].min())
print(df['Age'].max())
print(df['Age'].mean())