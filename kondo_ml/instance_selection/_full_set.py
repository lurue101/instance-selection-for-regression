import numpy as np
from sklearn.base import BaseEstimator

from kondo_ml.instance_selection.base import SelectorMixin


class FullSetSelector(SelectorMixin, BaseEstimator):
    """Nothing happens here. This class exist, such that the baseline of the full data set
    can be easily plugged in like other instance selection algorithms"""

    def __init__(self, subsize_frac=1):
        super().__init__(subsize_frac)

    def fit(self, X, y):
        self.nr_samples = X.shape[0]
        self.labels = np.ones(self.nr_samples, dtype="int8")
        self.scores = np.ones(self.nr_samples, dtype="int8")
        return self

    def predict(self, X, y):
        # just take the whole set
        self.labels = np.ones(self.nr_samples, dtype="int8")
        return self.labels
