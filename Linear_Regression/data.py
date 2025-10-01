import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from lr import LinearRegression

# Generate a synthetic regression dataset
X, y = make_regression(
    n_samples=300,     # number of samples
    n_features=1,      # number of input features
    noise=20,          # adds noise
    random_state=42
)

# Split the dataset into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Linear Regression model
linear_regressor =LinearRegression()
linear_regressor.fit(X_train,y_train)

# Make predictions
predictions=linear_regressor.predict(X_test)
# Evaluate the model
print("RSME:",rsme(predictions,y_test))

# Plotting the results
plt.figure(figsize=(8,10))
plt.scatter(X_test, y_test, color='blue', label='True Values', alpha=0.5)
plt.scatter(X_test, predictions, color='red', label='Predictions', alpha=0.5)
plt.plot(X_test, predictions, color='red', linewidth=2)
plt.xlabel('X feature')
plt.ylabel('Target y')  
plt.title('Linear Regression: True Values vs Predictions')
plt.legend()
plt.show()  
