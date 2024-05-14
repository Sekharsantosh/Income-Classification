# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 10:35:13 2023

@author: SANTOSH
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

a=[i for i in 'abcdefghijklmno']
data=pd.read_csv(r"C:\cars123.csv\AIML classesTHUB\income.csv",
                 names=a)
print(data.columns)
##a=np.arange(1,151)
##data['new']=a

print(data.shape)
#print(data)
print(data.head())

print(data.isna().sum())

data['b'].value_counts()
data['b'].replace(to_replace=' ?',value=' Private',inplace=True)

data['g'].value_counts()
data['g'].replace(to_replace=' ?',value=' Armed-Forces',inplace=True)

data['n'].value_counts()
data['n'].replace(to_replace=' ?',value=' United-States',inplace=True)

s='bdfghijn'
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in s:
    data[i]=le.fit_transform(data[i])

x=np.array(data.iloc[ : , :-1])##[0,1,2,3,5]])
y=data.iloc[ : ,-1].values
print(x.shape,y.shape)


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.4,random_state=1)

print(xtrain.shape,ytrain.shape)
print(xtest.shape,ytest.shape)


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(xtrain,ytrain)


ypred=model.predict(xtest)

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)

print(model.predict([[]]))
