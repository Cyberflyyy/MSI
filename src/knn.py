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

    def _kneighbors_indices(self, X, batch_size=256):
        """Zwraca indeksy k najbliższych sąsiadów w batchach (oszczędność pamięci)."""
        n = X.shape[0]
        indices = np.empty((n, self.k), dtype=np.intp)
        for start in range(0, n, batch_size):
            batch = X[start:start + batch_size]
            dist_batch = distance.cdist(batch, self.X_train, 'euclidean')
            indices[start:start + batch_size] = dist_batch.argsort()[:, :self.k]
        return indices

    def predict(self, X):
        X = X.toarray() if sp.issparse(X) else np.array(X)
        neighbors = self._kneighbors_indices(X)

        y_pred = []
        for line in neighbors:
            closest = np.unique(self.y_train[line], return_counts=True)
            most_popular = np.argmax(closest[1])
            y_pred.append(closest[0][most_popular])

        return np.array(y_pred)

    def predict_proba(self, X):
        X = X.toarray() if sp.issparse(X) else np.array(X)
        neighbors = self._kneighbors_indices(X)

        proba = []
        for line in neighbors:
            p1 = float(np.sum(self.y_train[line] == 1)) / self.k
            proba.append([1.0 - p1, p1])

        return np.array(proba)
