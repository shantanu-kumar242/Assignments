#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 09:28:42 2019

@author: shantanu
"""
#Import libabaries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

#Import Data
df=pd.read_csv("/home/shantanu/Downloads/sonar.all-data",sep=",", header=None)

#separating independent and dependent variables
X=df.iloc[ : , :60]
Y=df.iloc[ : , 60]

#Splitting data
x_train,x_test,y_train,y_test= train_test_split(X,Y,test_size=0.3,random_state=42)

#Decision Tree Model
max_depths = np.linspace(1, 32, 32, endpoint=True)
for max_depth in max_depths:
    dt=DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(x_train, y_train)
    dt_y_pred=dt.predict(x_test)
    data_accuracy_dt=accuracy_score(y_test,dt_y_pred)
    print(max_depth, data_accuracy_dt)

#optimum max_depth after hyperparameter tuning is max_depth=11

#Random Forest Model
n_est = np.linspace(51, 250, 200 , endpoint=True)
for n_estm in n_est:
    rf=RandomForestClassifier(n_estimators=int(n_estm), max_depth=11)
    rf.fit(x_train, y_train)
    rf_y_pred=rf.predict(x_test)
    data_accuracy_rf=accuracy_score(y_test,rf_y_pred)
    print(n_estm,data_accuracy_rf)

#optimum n_estimator after hyperparameter tuning is n_estimator=69


#SVM Model
gam=[0.1,1,10,100]
cs=[0.1,1,10,100]
for g in gam:
    for c in cs:
        svm= SVC(gamma=g,C=c)
        svm.fit(x_train, y_train)
        svm_y_pred=svm.predict(x_test)
        data_accuracy_svm=accuracy_score(y_test,svm_y_pred)
        print(g,c,data_accuracy_svm)

#optimum parameter for SVC is gamma=1 and C=1


#comparing models we find that random forest gives accuracy of 0.92, decision tree gives accuracy of 0.79 and svc gives accuracy of 0.92 at gamma=1 and C=10
#SVC works best though decision tree gave same accuracy but for each run on same hyperparameter decision tree gives different accuracy.



