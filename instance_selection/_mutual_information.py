from mutual_info.mutual_info import mutual_information

# from sklearn.feature_selection import mutual_info_regression
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors

from prism_kondo.instance_selection.base import SelectorMixin
from prism_kondo.utils import normalize_array


class MutualInformationSelector(SelectorMixin, BaseEstimator):
    def __init__(self, alpha=0.05, nr_of_neighbors=6, subsize_frac=1):
        super().__init__(subsize_frac)
        self.k = nr_of_neighbors
        self.alpha = alpha

    def fit(self, X, y):
        self.nr_of_samples = X.shape[0]
        self.labels = np.ones(self.nr_of_samples, dtype="int8") * -1
        self.scores = np.zeros(self.nr_of_samples, dtype="float32")
        return self

    def predict(self, X, y):
        subset_mask = np.zeros(self.nr_of_samples, dtype="bool")  # 1
        dict_neighbors = self.find_neighbors(X)  # 2,3 sample itself already excluded
        mutual_info = np.zeros(self.nr_of_samples)
        for i in range(self.nr_of_samples):
            X_without_i = np.delete(X, i, axis=0)
            y_without_i = np.delete(y, i, axis=0)
            mutual_info[i] = mutual_information(
                (X_without_i, y_without_i.reshape(-1, 1)), self.k
            )
        mutual_info = normalize_array(mutual_info)
        for i in range(self.nr_of_samples):
            c_diff = 0
            for j in range(self.k):
                jth_neighbor_of_xi = dict_neighbors[i][j]
                diff = mutual_info[i] - mutual_info[jth_neighbor_of_xi]
                if diff > self.alpha:
                    c_diff += 1
            self.scores[i] = (c_diff - self.k) * -1
            if c_diff < self.k:
                subset_mask[i] = True
        self.labels[subset_mask] = 1
        return self.labels

    def find_neighbors(self, X):
        nbrs = NearestNeighbors(n_neighbors=self.k + 1, algorithm="auto").fit(X)
        dict_neighbors = {}
        for i in range(self.nr_of_samples):
            investigated_instance = X[i, :].reshape(1, -1)
            indices = nbrs.kneighbors(investigated_instance, return_distance=False)
            indices = indices[0, 1:]
            dict_neighbors[i] = indices
        return dict_neighbors
