# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Read Data: Load the dataset (50_Startups.csv) using Pandas.
2.Preprocess Data: Extract feature matrix X (independent variables) and target variable y. Convert them to numerical format if needed.
3.Feature Scaling: Use StandardScaler to normalize both X and y for better convergence.
4.Initialize Parameters: Add a bias term (column of ones) to X, and initialize theta (weights) to zeros.
5.Gradient Descent: Iterate for num_iters, compute predictions, errors, and update theta using the gradient descent formula.
6.Prediction: Scale the new input data, apply the learned model (theta), and compute the predicted output.
7.Inverse Transform: Convert the scaled prediction back to its original scale using scaler.inverse_transform().

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: PRIDEESH M
RegisterNumber:   212223040154
 

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions - y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())
X=(data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_scaled=scaler.fit_transform(X1)
Y1_scaled=scaler.fit_transform(y)
print(X1_scaled)
print(Y1_scaled)
theta=linear_regression(X1_scaled,Y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")




*/
```

## Output:
![image](https://github.com/user-attachments/assets/6d9e87c7-a334-4643-8456-e9912c19b8d2)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
