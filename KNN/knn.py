
import numpy as np
from collections import Counter
def distance(x1,x2):
    distance = np.sqrt((np.sum(x1-x2)**2))
    return distance
class KNN:
    def __init__(self,k=3):
        self.k = k
        
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self,X):
        predictions=[]
        for x1 in X:
            
            distances =[]
            for x2 in self.X_train:
                dist = distance(x1,x2)
                distances.append(dist)
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices] 
            # majority vote
            predictions.append(Counter(k_nearest_labels).most_common()[0][0])
        return predictions
        

        
        

