import numpy as np
def accuracy(y_pred,y):
    return np.sum(y_pred==y)/len(y)
def precision(y_pred,y):
    true_positives = np.sum((y_pred==1) & (y==1))
    predicted_positives = np.sum(y_pred==1)
    if predicted_positives == 0:
        return 0.0
    return true_positives/predicted_positives
def recall(y_pred,y):
    true_positives = np.sum((y_pred==1) & (y==1))
    actual_positives = np.sum(y==1)
    if actual_positives == 0:
        return 0.0
    return true_positives/actual_positives
def f1_score(y_pred,y):
    prec = precision(y_pred,y)
    rec = recall(y_pred,y)
    if prec + rec == 0:
        return 0.0
    return 2*(prec*rec)/(prec+rec)
def mean_squared_error(y_pred,y):
    return np.mean((y_pred-y)**2)
def rsme(y_pred,y):
    return np.sqrt(mean_squared_error(y_pred,y))

