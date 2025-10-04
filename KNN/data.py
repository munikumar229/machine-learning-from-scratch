import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from knn import *
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
X,y = load_iris(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

knn = KNN(k=3)
knn.fit(X_train,y_train)
predictions = knn.predict(X_test)
print("Accuracy:",accuracy(predictions,y_test)) 


plt.figure(figsize=(8,10))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='o', label='True Labels', alpha=0.5)
plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions, cmap='coolwarm', marker='x', label='Predicted Labels', alpha=0.7)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KNN: True vs Predicted Labels')
plt.legend()
plt.show()
