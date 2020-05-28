# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 21:42:40 2020

@author: Rai Kanwar Taimoor
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from prettytable import PrettyTable

def Result_Table(table_col1, table_col2):
    table = PrettyTable()
    table.add_column("Actual Label", table_col1)
    table.add_column("Predicted Value", table_col2)
    return table
def gradientdescent(X_train, Y_train, theta, alpha, iterations):
    for _ in range(iterations):
        theta = theta - ((alpha/m) * X_train.T @ (sigmoid(X_train @ theta) - Y_train))
    return theta
def sigmoid(x):
  return 1/(1+np.exp(-x))

def costFunction(theta, X_train, Y_train):
    J = (-1/m) * np.sum(np.multiply(Y_train, np.log(sigmoid(X_train @ theta))) 
        + np.multiply((1-Y_train), np.log(1 - sigmoid(X_train @ theta))))
    return J

#intialization 
df = pd.read_csv('IrisSepal.txt', delim_whitespace=True, names=('X1', 'X2', 'Y'))
x=df.iloc[:,[0,1]]
y=df.iloc[:,2]
zero = df.loc[y =="Iris-setosa"]
one = df.loc[y == "Iris-virginica"]
for i in range(len(df)) : 
  y[i]=df.loc[i, "Y"]
for i in range(len(y)) :
   y=np.where(y=="Iris-setosa",0, y)  
for i in range(len(y)) :
   y=np.where(y=="Iris-virginica",1, y)    
y = y[:, np.newaxis]

x = np.c_[np.ones((x.shape[0], 1)), x]
theta = np.zeros((x.shape[1], 1))
y = y.astype('float64') 


# Spiliting Data 67-33 ratio as said by sir
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.33,random_state=0)

m=len(Y_train)
iterations = 1500
alpha = 0.01

#training of model
theta = gradientdescent(X_train, Y_train, theta, alpha, iterations)
J = costFunction(theta, X_train, Y_train)
print("cost: ",J)

hypothesis_train=sigmoid(X_train @ theta)


plt.scatter(zero.iloc[:, 0], zero.iloc[:, 1], s=10, label='Iris-setosa')
plt.scatter(one.iloc[:, 0], one.iloc[:, 1], s=10, label='Iris-virginica')
plt.legend()

x1_vals = np.linspace(X_train[:,[1]].min(), X_train[:,[1]].max(), iterations)
x2_vals = (-theta[0, 0] - (theta[1, 0] * x1_vals)) / theta[2, 0]

plt.plot(x1_vals,x2_vals,color='black')
plt.show()


# predicted values
predicted=[0]*len(Y_test)

for i in range(len(Y_test)):
    hypothesis_test=sigmoid(X_test[i] @ theta)
    if( hypothesis_test <= 0.5):
        predicted[i]=0
    else:
        predicted[i]=1

#to print the table of actual vs predicted value
print(Result_Table(Y_test,predicted))

from sklearn.metrics import confusion_matrix
 
results = confusion_matrix(Y_test,predicted)
print("Confustion matrix \n",results)

# to print the values suggested by sir 
# didnt print false negative cuz already printed in confussion matrix above it would be redundant otherwise

from sklearn.metrics import precision_score ,recall_score , f1_score,accuracy_score
precision = precision_score(Y_test,predicted, average='binary')
print('Precision: %.3f' % precision)

recall = recall_score(Y_test,predicted, average='binary')
print('recall: %.3f' % recall)

F1_score = f1_score(Y_test,predicted, average='binary')
print('f1_score: %.3f' % F1_score)

accuracy_score = accuracy_score(Y_test,predicted)
print('accuracy_score: %.3f' % accuracy_score)

print("theeta:", theta)

