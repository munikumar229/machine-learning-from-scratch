import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from svm import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

X,y = make_blobs(n_samples=300,
                 n_features=2,
                 centers=2,
                 cluster_std=1.05,
                 random_state=40
                 )
y = np.where(y<=0,-1,1)
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)

svm = SVM()
svm.fit(X_train,y_train)
predictions = svm.predict(X_test)

print("Accuracy:", accuracy(predictions, y_test))
print("Precision:", precision(predictions, y_test))
print("Recall:", recall(predictions, y_test))
plot_svm(X, y, svm)
