# -*- coding: utf-8 -*-
"""
Created on Mon May 11 05:41:26 2020

@author: Rai Kanwar Taimoor
"""

import numpy as np
import pandas as pd
from prettytable import PrettyTable

# Data Initialization and variable separation
df = pd.read_csv('spambase.data',delimiter=',',header=None)
df = df.dropna()
x=df.loc[:, 0:56]
y=df.iloc[:,-1]

# Spiliting Data randomly  as said by sir
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.33,random_state=0)
 
#logistic regression with Regression 
"""
C is the inverse of lamda thus the lesser the c value the greater and more regularized data
"""

from sklearn import svm
#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel
#Train the model using the training sets
clf.fit(X_train, Y_train)
#Predict the response for test dataset
predicted = clf.predict(X_test)
#to print the table of actual vs predicted value
noob=predicted.astype(str)
y = Y_test[:, np.newaxis]
y_p = noob[:, np.newaxis]
table = PrettyTable()
table.add_column("Actual Label",y)
table.add_column("Predicted Value",y_p)
print(table)

# to print the values 
# didnt print false negative cuz already printed in confussion matrix above it would be redundant otherwise
from sklearn.metrics import confusion_matrix
results = confusion_matrix(Y_test,predicted)
print("Confustion matrix \n",results)
from sklearn.metrics import precision_score 
precision = precision_score(Y_test,predicted, average='micro')
print('Precision: %.3f' % precision)
from sklearn.metrics import recall_score
recall = recall_score(Y_test,predicted, average='micro')
print('recall: %.3f' % recall)
from sklearn.metrics import f1_score
F1_score = f1_score(Y_test,predicted, average='micro')
print('f1_score: %.3f' % F1_score)
from sklearn.metrics import accuracy_score
accuracy_score = accuracy_score(Y_test,predicted)
print('accuracy_score: %.3f' % accuracy_score)