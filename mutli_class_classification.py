# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:26:10 2020

@author: Rai Kanwar Taimoor
"""

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.linear_model import LogisticRegression

def Result_Table(table_col1, table_col2):
    table = PrettyTable()
    table.add_column("Actual Label", table_col1)
    table.add_column("Predicted Value", table_col2)
    return table

# Initialization
df = pd.read_csv('iris.data',sep=',',header =None,names=['x1','x2','x3','x4','y'])

df=df.dropna()
x=df.iloc[:,[0,1,2,3]]
y=df.iloc[:,4]

# Spiliting Data 67-33 ratio as said by sir
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.33,random_state=0)
 

#to avoid regularization that is by default these parameters are passed so now there is no regularization
model = LogisticRegression(solver = 'lbfgs',penalty='l2', C = 2e10) #to Avoid Regularization !!!
model.fit(X_train, Y_train)
predicted = model.predict(X_test)

#to print the table of actual vs predicted value
noob=predicted.astype(str)
y = Y_test[:, np.newaxis]
y_p = noob[:, np.newaxis]
print(Result_Table(y,y_p))


# to print the values suggested by sir 
# didnt print false negative cuz already printed in confussion matrix above it would be redundant otherwise
from sklearn.metrics import confusion_matrix
results = confusion_matrix(Y_test,predicted)
print("Confustion matrix \n",results)


from sklearn.metrics import precision_score ,recall_score , f1_score,accuracy_score
precision = precision_score(Y_test,predicted, average='micro')
print('Precision: %.3f' % precision)

recall = recall_score(Y_test,predicted, average='micro')
print('recall: %.3f' % recall)

F1_score = f1_score(Y_test,predicted, average='micro')
print('f1_score: %.3f' % F1_score)

accuracy_score = accuracy_score(Y_test,predicted)
print('accuracy_score: %.3f' % accuracy_score)