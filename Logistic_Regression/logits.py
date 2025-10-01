import numpy as np
from utils import *
class LogisticRegression:
    def __init__(self,weights = None,bias = None,lr=0.01,iter = 10000):
         self.weights = weights
         self.bias = bias 
         self.lr = lr 
         self.iter = iter
    def fit(self,X,y):
         
        n_samples,feats = X.shape
        n_classes = len(np.unique(y))
        y = np.eye(n_classes)[y.reshape(-1)]
         #one hot encoding
         #print(y)
         #initializing weights and bias
        self.weights = np.zeros((feats,n_classes))
        self.bias = np.zeros((1,n_classes))
        for i in range(self.iter):
            y_pred = self.softmax(X @ self.weights + self.bias)
            dw = (1/n_samples) * (X.T @ (y_pred - y))
            db = (1/n_samples) * np.sum((y_pred - y), axis=0, keepdims=True)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            if i % 1000 ==0:
                loss = cross_entropy(y_pred,y)
                print(f"cross_entropy at {i} : {loss}")
        return self

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # for stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


         
    def predict(self,X):
        y_pred = X @ self.weights + self.bias
        prob = self.softmax(y_pred)
        return np.argmax(prob,axis = 1)