import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
import matplotlib.pyplot as plt
class NaiveBayes:

    def fit(self,X,y):
        n_samples,feats = X.shape
        self.classes = len(np.unique(y))
        self._classes = np.unique(y)
        self.mean = np.zeros((self.classes,feats),dtype=np.float64)
        self.var = np.zeros((self.classes,feats),dtype=np.float64)
        self.priors = np.zeros(self.classes,dtype = np.float64)

        for idx,c in enumerate(np.unique(y)):
            X_c = X[y==c]
            self.mean[idx,:] = np.mean(X_c,axis=0)
            self.var[idx,:] = np.var(X_c,axis=0)
            self.priors[idx] = X_c.shape[0]/float(n_samples)
        return self
    def predict(self,X):
        y_pred = [ self._predict(x) for x in X]
        return np.array(y_pred)
    def _predict(self,x):
        posteriors=[]
        for idx,c in enumerate(np.unique(y)):
            prior = np.log(self.priors[idx])
            posterior = np.sum(np.log(self._guassian(idx,x)))
            posterior = posterior+prior
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]
    def _guassian(self,idx,x):
        mean = self.mean[idx]
        var = self.var[idx]

        numerator = np.exp(-((x-mean)**2)/(2*var))
        denominator = np.sqrt(2*np.pi*var)
        return numerator/denominator





X,y = make_classification(n_samples=1000,
                          n_features=20,
                          n_classes=2,
                          random_state=42)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
naive = NaiveBayes()
naive.fit(X_train,y_train)
y_pred = naive.predict(X_test)
print("accuracy : ",accuracy(y_pred,y_test))
print("precision : ",precision(y_pred,y_test))
print("recall : ",recall(y_pred,y_test))    

plt.figure(figsize=(10,7))  
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='bwr', alpha=0.7)    
plt.title("Naive Bayes Classification Results")
plt.show()