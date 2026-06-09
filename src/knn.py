import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial import distance
import scipy.sparse as sp


class KNearestNeighbors(ClassifierMixin, BaseEstimator):

    def __init__(self, k=5):
        self.k = k

    def fit(self, X_train, y_train):
        # cdist wymaga gęstej macierzy
        self.X_train = X_train.toarray() if sp.issparse(X_train) else np.array(X_train)
        self.y_train = np.array(y_train)
        return self

    def predict(self, X):
        X = X.toarray() if sp.issparse(X) else np.array(X)
        distances = distance.cdist(X, self.X_train, 'euclidean')
        distances_sort = distances.argsort()[:, :self.k]

        y_pred = []

        for line in distances_sort:
            closest = np.unique(self.y_train[line], return_counts=True)
            most_popular = np.argmax(closest[1])
            y_pred.append(closest[0][most_popular])

        return np.array(y_pred)

    def predict_proba(self, X):
        X = X.toarray() if sp.issparse(X) else np.array(X)
        distances = distance.cdist(X, self.X_train, 'euclidean')
        distances_sort = distances.argsort()[:, :self.k]

        proba = []
        for line in distances_sort:
            p1 = float(np.sum(self.y_train[line] == 1)) / self.k
            proba.append([1.0 - p1, p1])

        return np.array(proba)
