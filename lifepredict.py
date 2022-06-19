from ctypes import sizeof
from matplotlib.pyplot import plot
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import make_column_transformer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from model_plot import plot_learning_curve

#read csv file
df = pd.read_csv('LifeExpectancyData.csv')
df = df.dropna()
y = df.get('Life expectancy ')
X = df.iloc[:,np.r_[0:3,4:22]]
#print()
print(df.info())
print(X.info())

#X_numeric = X.iloc[:,np.r_[0,1,8,9,10,11,12,15,17,19:25,26,27]]
#print(X_numeric.info())
#X_train, X_test, y_train,y_test = train_test_split(X_numeric,y,test_size=0.3, random_state=42)
#encode data
enc = OneHotEncoder()
transformed =enc.fit_transform(X[['Status']])
print(enc.categories_)
X[enc.categories_[0]] = transformed.toarray()

#drop nominal data
X= X.drop(['Country','Year','Status'], axis=1)
print(X.head())

#nomaliza data
normalizer = StandardScaler()
X=pd.DataFrame(normalizer.fit_transform(X), columns=X.columns)
#print(y)

#splitting dataset
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=42)

#create model parameters
linear_model = LinearRegression(fit_intercept=True)
linear_model.fit(X_train,y_train)
print(linear_model.intercept_)
print(linear_model.score(X_test,y_test))

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

fig, axes = plt.subplots(3, 3, figsize=(10, 15))
lg_estimator = LinearRegression()
mlp_estimator = MLPRegressor(hidden_layer_sizes=(50,3),max_iter=1500)
plot_learning_curve(lg_estimator,"Learning Curve",X_train,y_train,axes=axes[:, 0],ylim=(0.3,1.01))
plot_learning_curve(mlp_estimator,"Learning Curve",X_train,y_train,axes=axes[:, 1],ylim=(0.7,1.01))


#polynomial
poly = PolynomialFeatures(2)
X_train=pd.DataFrame(poly.fit_transform(X_train))
print(X_train.info())
plot_learning_curve(lg_estimator,"Learning Curve",X_train,y_train,axes=axes[:, 2],ylim=(0.3,1.01))

plt.show()