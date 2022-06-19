from matplotlib.pyplot import plot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from model_plot import plot_learning_curve

#read csv file
df = pd.read_csv('automobile.csv')

y = df.get('price')
X = df.iloc[:,np.r_[0:24,25:29]]
#print()
print(df.info())
print(X.info())

X_numeric = X.iloc[:,np.r_[0,1,8,9,10,11,12,15,17,19:25,26,27]]
print(X_numeric.info())
X_train, X_test, y_train,y_test = train_test_split(X_numeric,y,test_size=0.3, random_state=42)

#create model parameters
linear_model = LinearRegression(fit_intercept=True)
linear_model.fit(X_train,y_train)
print(linear_model.intercept_)
print(linear_model.score(X_test,y_test))



fig, axes = plt.subplots(3, 2, figsize=(10, 15))
plot_learning_curve(LinearRegression(),"Learning Curve",X_numeric,y,axes=axes[:, 0],ylim=(0.7,1.01))
plt.show()