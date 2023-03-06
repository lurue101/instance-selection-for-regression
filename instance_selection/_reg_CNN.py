import numpy as np

from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors

from ..model import train_lr_model
from .base import SelectorMixin


class RegCnnSelector(SelectorMixin, BaseEstimator):
    def __init__(self, alpha=0.25, nr_of_neighbors=7, subsize_frac=1):
        super().__init__(subsize_frac=subsize_frac)
        self.k = nr_of_neighbors
        self.alpha = alpha

    def fit(self, X, y):
        self.nr_of_samples = X.shape[0]
        self.labels = np.ones(self.nr_of_samples, dtype="int8") * -1
        self.scores = np.zeros(self.nr_of_samples, dtype="float32")
        return self

    def predict(self, X, y):
        subset_mask = np.zeros(self.nr_of_samples, dtype="bool")  # 1
        subset_mask[0] = True  # 2
        nn_mask = np.ones(self.nr_of_samples, dtype="bool")
        for i in range(1, self.nr_of_samples):
            if nn_mask.sum() <= self.k:
                self.labels[subset_mask] = 1
                return self.labels
            investigated_instance = X[i, :].reshape(1, -1)
            model = train_lr_model(X[subset_mask, :], y[subset_mask])
            y_pred = model.predict(investigated_instance)  # 3
            # As the closest neighbor is always the instance itself, we add one neighbor and ignore the 0 index
            nbrs = NearestNeighbors(n_neighbors=self.k + 1, algorithm="auto").fit(
                X[nn_mask, :]
            )
            indices = nbrs.kneighbors(investigated_instance, return_distance=False)
            indices = indices[0, 1:]  # 5
            theta = self.alpha * np.std(y[indices])  # 6
            y_true = y[i]
            self.scores[i] = (theta - np.abs(y_true - y_pred)) * -1
            if np.abs(y_true - y_pred) > theta:  # 7
                subset_mask[i] = True  # 8
                nn_mask[i] = False  # 9
        self.labels[subset_mask] = 1
        return self.labels
