import numpy as np

from sklearn.base import BaseEstimator
from sklearn.neighbors import LocalOutlierFactor

from prism_kondo.instance_selection.base import SelectorMixin


class LOFSelector(SelectorMixin, BaseEstimator):
    def __init__(self, nr_of_neighbors=2):
        self.k = nr_of_neighbors

    def fit(self, X, y):
        self.nr_of_samples = X.shape[0]
        self.labels = np.ones(self.nr_of_samples, dtype="int8") * -1
        self.sklearn_obj = LocalOutlierFactor(n_neighbors=self.k, novelty=False)
        return self

    def predict(self, X, y):
        self.labels = self.sklearn_obj.fit_predict(X)
        self.scores = self.sklearn_obj.negative_outlier_factor_
        return self.labels
