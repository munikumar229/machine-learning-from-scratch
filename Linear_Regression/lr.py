
import numpy as np
import utils
class LinearRegression():
    def __init__(self,lr = 0.001,weights=None,bias = None,iter = 10000):
        self.lr = lr
        self.weights= weights
        self.bias = bias 
        self.iter = iter

    # training the model
    def fit(self,X,y):
        n_samples,feats = X.shape #number of samples and features
        # initialize weights and bias
        self.weights = np.zeros(feats)
        self.bias = 0.0
        
        for i in range(self.iter):
            y_pred = X @ self.weights + self.bias 
            
            if i % 1000==0:
                loss= utils.rsme(y_pred,y)
                print(f"loss in {i} = {loss}")

            dw = (1/n_samples)*(X.T @ (y_pred-y))   # weight gradient
            db = (1/n_samples)*np.sum(y_pred-y)     # bias gradient
            self.weights-=self.lr*dw                # update weights
            self.bias-=self.lr*db
        return self

            
    # predictions
    def predict(self,X):
        y_pred = X @ self.weights + self.bias
        return y_pred

        
