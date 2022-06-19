import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

pd.set_option('display.max_columns',None)
df = pd.read_csv('automobile.csv')
#print(df)

#descriptive statistics
print(df.info())

#generate boxplot
#sns.boxplot(x="num-of-cylinders",y="price",data=df)
#plt.show()

#scatter plot
#plt.scatter(df['wheel-base'],df['price'])
#plt.xlabel('wheel-base')
#plt.ylabel('Price')
#plt.show()

#histogram
#count,bin_edges = np.histogram(df['peak-rpm'])
#df['peak-rpm'].plot(kind='hist',xticks=bin_edges)
#plt.xlabel('Value of peak rpm')
#plt.ylabel('Number of cars')
#plt.grid()
#plt.show()

#correlation heatmap
correlation_matrix = df.corr()
#sns.heatmap(correlation_matrix, annot=False)
#plt.show()

#regression plot (univariate analysis)
sns.regplot(x="wheel-base",y='price',data=df)
plt.show()
