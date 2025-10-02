import numpy as np
import matplotlib.pyplot as plt
class SVM:
    def __init__(self,lr =0.001,weights = None,bias = None,alpha = 0.01,iter = 1000):
        self.lr=lr 
        self.weights = weights
        self.bias = bias 
        self.alpha = alpha
        self.iter = iter
        self.support_vectors_ = None

    def fit(self,X,y):
        n_samples , feats = X.shape
        self.weights = np.zeros(feats)
        self.bias = 0.0
        y = np.where(y<=0,-1,1)
        support_vector_indices = []
        for _ in range(self.iter):
            for idx,x_i in enumerate(X):
                margin = y[idx] * (np.dot(self.weights,x_i)+self.bias)
                condition = margin >=1
                if condition:
                    dw = 2 * self.alpha * self.weights
                    self.weights -= self.lr * dw
                else:
                    db = -y[idx]
                    self.bias -= self.lr * db
                    dw = 2*self.alpha*self.weights - np.dot(y[idx],x_i)
                    self.weights -= self.lr * dw
                # Track support vectors (margin close to 1)
                if abs(margin - 1) < 1e-2:
                    support_vector_indices.append(idx)
        # Store unique support vectors
        self.support_vectors_ = X[np.unique(support_vector_indices)]
        return self
    def predict(self,X):
        return np.sign(np.dot(X,self.weights)+self.bias)

    def decision_function(self, X):
        """Returns the distance to the decision boundary for each sample."""
        return np.dot(X, self.weights) + self.bias
def plot_svm(X, y, svm):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7)
    # Plot support vectors
    if hasattr(svm, 'support_vectors_') and svm.support_vectors_ is not None:
        plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')
    # Plot decision boundary
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = svm.decision_function(xy).reshape(XX.shape)
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    plt.legend()
    plt.show()