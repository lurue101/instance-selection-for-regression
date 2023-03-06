import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from .base import SelectorMixin


class RandomSelector(SelectorMixin, BaseEstimator):
    def __init__(self, subsize_frac):
        super().__init__(subsize_frac)

    def fit(self, X, y=None):
        self.nr_samples = X.shape[0]
        self.subsize_int = self.calc_subset_sizeint(self.nr_samples)
        self.labels = np.ones(self.nr_samples, dtype="int8") * -1
        self.scores = np.random.uniform(-1, 1, self.nr_samples)
        return self

    def predict(self, X, y=None):
        idx_sorted = np.argsort(self.scores)
        idx_selected = idx_sorted[-self.subsize_int :]
        self.labels[idx_selected] = 1
        return self.labels
