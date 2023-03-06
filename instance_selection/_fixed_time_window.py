import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator

from .base import SelectorMixin


class FixedTimeSelector(BaseEstimator, SelectorMixin):
    def __init__(self, subsize_frac=0.5):
        super().__init__(subsize_frac)

    def fit(self, X, y=None):
        """
        The time vector needs to be that last column in the X array
        Parameters
        ----------
        X
        y

        Returns
        -------

        """
        self.labels = np.ones(X.shape[0], dtype="int8") * -1
        self.nr_of_samples_to_pick = self.calc_subset_sizeint(X.shape[0])
        self.time_vector = pd.to_datetime(X[:, -1]).to_numpy()
        self.scores = np.linspace(-1, 1, X.shape[0], dtype="float32")

        return self

    def predict(self, X, y=None):
        sorted_idxs = np.argsort(self.time_vector)
        subset_idxs = sorted_idxs[-self.nr_of_samples_to_pick :]
        self.labels[subset_idxs] = 1
        return self.labels
