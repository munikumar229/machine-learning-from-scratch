
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from logits import LogisticRegression

X,y = load_digits(return_X_y=True)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

log_reg = LogisticRegression(lr=0.001,iter=10000)
log_reg.fit(X_train,y_train)
predictions = log_reg.predict(X_test)
print("cross entropy:",cross_entropy(np.eye(10)[predictions.reshape(-1)],np.eye(10)[y_test.reshape(-1)]))
print("Accuracy:",accuracy(predictions,y_test))
print("Precision:",precision(predictions,y_test))
print("Recall:",recall(predictions,y_test)) 

plt.figure(figsize=(10,7))
plt.imshow(X_test[7].reshape(8,8),cmap='gray')
plt.title(f"Predicted : {predictions[7]}, Actual : {y_test[7]}")
plt.show()
