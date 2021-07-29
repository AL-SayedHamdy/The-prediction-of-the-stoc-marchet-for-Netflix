#Importing the libraries
import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
import pandas_datareader.data as data
from pandas_datareader.data import DataReader
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
#Importing the dataset
df = pd.read_csv('nflx.csv')
#The linear model
'''
#Making the lists
days = list()
adj_close_prices = list()
#Locate the data from the dataset
df_days = df.loc[:, 'Date']
df_adj_close = df.loc[:, 'Adj Close']
#The dependent and independent variables
for day in df_days:
    days.append([int(day.split('-')[2])])
for adj_close_price in df_adj_close:
    adj_close_prices.append(float(adj_close_price))
#3 SVR models
lin_svr = SVR(kernel=('linear'),)
lin_svr.fit(days, adj_close_prices)

poly_svr = SVR(kernel=('poly'), degree=2)
poly_svr.fit(days, adj_close_prices)

rbf_svr = SVR(kernel=('rbf'), gamma= 0.85)
rbf_svr.fit(days, adj_close_prices)
#Visualising the data
plt.scatter(days, adj_close_prices, color= 'black', label= 'Data')
plt.plot(days, lin_svr.predict(days), color = 'red', label= 'Linear')
plt.plot(days, poly_svr.predict(days), color = 'green', label= 'Poly')
plt.plot(days, rbf_svr.predict(days), color = 'blue', label= 'RBF')
plt.xlabel('Days')
plt.ylabel('Adj close price')
plt.legend()
plt.show()
'''
#Transforming the index column to the date
df = df.set_index(pd.DatetimeIndex(df['Date'].values))
df.index.name = 'Date'
#Delete the data column from the dataset
df = df.drop(columns = ['Date'])
#Adding the "Win" column to the dataset
df['Win'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
#Split the dataset
X = df.iloc[:, :6].values
Y = df.iloc[:, -1]
#The train and the test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
#The algo (Decision Tree)
tree = DecisionTreeClassifier().fit(X_train, Y_train)
Pred = tree.predict(X_test)
#KNN
knn = KNeighborsClassifier(n_neighbors= 1)
knn.fit(X_train, Y_train)
knn.predict(X_test)
#The report
rep = classification_report(Pred, Y_test)
cm = confusion_matrix(Pred, Y_test)
#Visualisation
sns.pairplot(df, hue= 'Win', palette='coolwarm')