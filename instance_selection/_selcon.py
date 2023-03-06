import numpy as np

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from prism_kondo.instance_selection.base import SelectorMixin
from prism_kondo.SELCON.datasets import get_data
from prism_kondo.SELCON.linear import Regression


class SelconSelector(SelectorMixin, BaseEstimator):
    def __init__(self, val_frac=0.1, subsize_frac=0.8):
        super().__init__(subsize_frac)
        self.val_frac = val_frac

    def fit(self, X, y):
        indices = np.arange(X.shape[0])
        idx_train, idx_val = train_test_split(
            indices, test_size=self.val_frac, shuffle=False
        )
        self.idx_train = idx_train
        self.X_train, self.X_val, self.y_train, self.y_val = get_data(
            X[idx_train, :], X[idx_val, :], y[idx_train], y[idx_val]
        )
        self.labels = np.ones(X.shape[0], dtype="int8") * -1
        self.scores = np.ones(X.shape[0], dtype="int8") * -1
        return self

    def predict(self, X, y):
        reg = Regression()
        reg.train_model(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            fraction=self.subsize_frac,
        )
        subset_idxs = reg.return_subset()
        self.labels[self.idx_train[subset_idxs]] = 1
        self.scores = self.labels
        return self.labels
