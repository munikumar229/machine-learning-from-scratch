import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
class PCA:

    '''
    1. Standarize the data
    2. Covariance matrix
    3. Calculate eigen vectors and eigen values
    4. Choose best components sorting in desc order
    ''' 

    def __init__(self,n_components = 2):
        self.n_components = n_components
        self.components = None
        self.mean = None

        pass
    def fit(self,X):

        self.X = X
        self.mean = np.mean(X,axis =0)
        self.X -= self.mean
        cova = np.cov(X.T)
        eigen_vectors ,eigen_values = np.linalg.eig(cova)
        eigen_vectors = eigen_vectors.T

        idxs = np.argsort(eigen_values)[::-1]
        eigen_vectors = eigen_vectors[idxs]
        eigen_values = eigen_vectors[idxs]

        self.components = eigen_vectors[:self.n_components]

        return self

    def transform(self,X):
        self.X = X- self.mean
        return np.dot(X,self.components.T)

X,y = load_iris(return_X_y = True)
pca =PCA(n_components = 2)
pca.fit(X)
X_pca = pca.transform(X)

plt.scatter(X_pca[:,0],X_pca[:,1],c = y)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA on Iris Dataset")
plt.show()