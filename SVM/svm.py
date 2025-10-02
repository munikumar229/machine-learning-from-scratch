import numpy as np
class SVM:
    def __init__(self,lr =0.01,weights = None,bias = None,alpha = 0.001,iter = 1000):
        self.lr=lr 
        self.weights = weights
        self.bias = bias 
        self.alpha = alpha
        self.iter = iter

    def fit(self,X,y):
        n_samples , feats = X.shape
        self.weights = np.zeros(feats)
        self.bias = 0.0
        y = np.where(y==0,-1,1)
        for _ in iter:
            for idx,x_i in enumerate(X):
                condition = y[idx] * (np.dot(self.weights,x_i)+self.bias) >=1
                if condition:
                    dw = 2 * self.alpha * self.weights
                    self.weights -= self.lr * dw
                else:
                    db = -y[idx]
                    self.bias -= self.lr * db
                    dw = 2*self.alpha*self.weights - np.dot(y[idx],x_i)
                    self.weights -= self.lr * dw



        return self
    def predict(self,X):
        return np.sign(np.dot(X,self.weights)+self.bias)