import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
def activation(y):
    return np.where(y>0,1,0)
class Perceptron:
    def __init__(self,lr = 0.01,iter =1000):
        self.lr = lr 
        self.iter = iter
        self.weights = None
        self.bias = None

    def fit(self,X,y):
        n_samples,feats = X.shape
        self.weights = np.zeros(feats)

        self.bias = 0.0
        y = np.where(y>0,1,0)

        for _ in range(self.iter):
            for idx, x in enumerate(X):
                f = np.dot(x,self.weights)+self.bias
                y_pred = activation(f) 

                # update weights and bias
                dw = (y[idx]-y_pred)*x
                self.weights+=self.lr * dw
                db = y[idx]-y_pred
                self.bias+=self.lr*db
        return self
    def predict(self,X):
        y_pred = activation(np.dot(X,self.weights)+self.bias)
        return y_pred


X,y = make_blobs(n_samples=300,
                 n_features=2,
                 centers=2,
                 cluster_std=1.05)
y=np.where(y>0,1,0)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print(X_test.shape)
perceptron = Perceptron()
perceptron.fit(X_train,y_train)


predictions = perceptron.predict(X_test)
print(predictions)
print(y_test)
print("Accuracy:",accuracy(predictions,y_test))
print("Precision:",precision(predictions,y_test))
print("Recall:",recall(predictions,y_test))
print("F1 Score:",f1_score(predictions,y_test))
print("Cross Entropy:",cross_entropy(predictions,y_test))

import matplotlib.pyplot as plt

# Plot training data
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', alpha=0.7)
x_min, x_max = plt.xlim()

# Compute decision boundary: w1*x + w2*y + b = 0  => y = (-b - w1*x)/w2
x_values = np.linspace(x_min, x_max, 100)
y_values = -(per.weights[0] * x_values + per.bias) / per.weights[1]

plt.plot(x_values, y_values, 'k--', label='Decision Boundary')
plt.legend()
plt.title("Perceptron Decision Boundary")
plt.show()
