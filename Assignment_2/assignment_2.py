import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("/home/shantanu/Downloads/winequality-white.csv", sep=";")
#print(df.head())
#df2=pd.read_csv("/home/shantanu/Downloads/winequality-red.csv", sep=";")
#print(df2.head())

X=df.drop(['quality'],axis=1)
Y=df['quality'].values.reshape((X.shape[0],1))
theta=np.zeros([1,12])
theta = theta.reshape((12,1))
one = np.ones((X.shape[0],1))
X = np.concatenate((one,X),axis=1)
    
def  cal_cost(theta,X,Y):
    m = len(Y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions-Y))
    return cost

def gradient_descent(X,Y,theta,learning_rate,iterations):
    m = len(Y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,12))
    for it in range(iterations):
        
        predictions = np.dot(X,theta)
        
        theta = theta -(1/m)*learning_rate*( X.T.dot((predictions - Y)))
        theta_history[it,:] =theta.T
        cost_history[it]  = cal_cost(theta,X,Y)
        
    return theta, cost_history, theta_history

learning_rate = 0.000001
iterations = 100000

t,c,th = gradient_descent(X,Y,theta,learning_rate,iterations)

ni = [i for i in range(iterations)]
plt.plot(ni,c)
