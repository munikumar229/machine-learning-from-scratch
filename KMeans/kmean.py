import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import make_blobs
'''
Step1 : intialize the centroids
Step2 : label the X based on cluster indiices
Step3 : Update the centroids
        1. Compute the new centroids as the mean of the points in each cluster
        2. Check for convergence (if centroids do not change)
Step4 : Repeat step 2 and step 3 until convergence or max iterations reached
'''


def eucleadian_distance(centroids,x):
    '''Calculate the Euclidean distance between a point and multiple centroids.'''
    return np.sqrt(np.sum((centroids-x)**2,axis =1))

def SSE(centroids,X,labels):
    '''Calculate the Sum of Squared Errors (SSE) for the clustering.'''
    sse = 0
    for i,centroid in enumerate(centroids):
        cluster_points = X[labels==i]
        sse += np.sum((cluster_points - centroid)**2)
    return sse

class KMeans:
    def __init__(self,centroids = None,k=3,iter = 1000):
        self.k = k 
        self.iter = iter
        self.centroids = centroids
    
    def predict(self,X):
        self.X = X
        random_idx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[random_idx]
        # self.centroids = np.random.uniform(np.amin(X,axis=0),np.amax(X,axis=0),size = (self.k,X.shape[1]))
        for _ in range(self.iter):
            y=[]
            for x in self.X:

                distances = eucleadian_distance(self.centroids,x)
                cluster_indic=np.argmin(distances)
                y.append(cluster_indic) 
            y=np.array(y)
            cluster_indices = []
            for i in range(self.k):
                indices = np.where(y == i)[0]
                cluster_indices.append(indices)
            cluster_centers = []
            for i ,indicies in enumerate(cluster_indices):
                if len(indicies)==0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indicies],axis=0))
            if np.linalg.norm(np.array(cluster_centers) - self.centroids) < 1e-5:
                break
            else:
                self.centroids = np.array(cluster_centers)
        return y
        


        

        

# X = np.random.randint(0,100,size = (100,2))
X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
k = [1,2,3,4,5,6,7,8]
loss = []
for i in k:
    knn=KMeans(k=i)
    y = knn.predict(X)
    loss.append(SSE(knn.centroids,X,y))

# Plotting the Elbow Curve
plt.figure(figsize=(8,5))
plt.plot(k, loss, 'bo-', linewidth=2, markersize=8, label='SSE Curve')

for i, val in enumerate(loss):
    plt.text(k[i], val, f'{val:.0f}', ha='center', va='bottom', fontsize=9)

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Sum of Squared Errors (SSE)")
plt.title("Elbow Method for Optimal k")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
elbow_k = k[np.argmin(np.diff(loss)) + 1]
plt.scatter(elbow_k, loss[k.index(elbow_k)], c='red', s=120, marker='X', label='Elbow Point')
plt.show()
# Using the optimal k to fit the model

knn=KMeans(k=elbow_k)
y = knn.predict(X)
plt.scatter(X[:,0],X[:,1],c=y,cmap='viridis')
plt.scatter(knn.centroids[:,0],knn.centroids[:,1],c='red',marker='X',s=200)
plt.show()
