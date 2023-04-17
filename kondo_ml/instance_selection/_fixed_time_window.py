import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from kondo_ml.instance_selection.base import SelectorMixin


class FixedTimeSelector(BaseEstimator, SelectorMixin):
    """Selects samples based on the recency of the sample"""

    def __init__(self, subsize_frac=0.5):
        super().__init__(subsize_frac)

    def fit(self, X, y=None):
        """
        The time vector needs to be that last column in the X array

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self
            Fitted estimator.
        """
        self.labels = np.ones(X.shape[0], dtype="int8") * -1
        self.nr_of_samples_to_pick = self.calc_subset_sizeint(X.shape[0])
        self.time_vector = pd.to_datetime(X[:, -1]).to_numpy()
        self.scores = np.linspace(-1, 1, X.shape[0], dtype="float32")

        return self

    def predict(self, X, y=None):
        """Predict the labels (1 use for training, -1 rejected) of X according to time vector present in the X
        array passed to the fit method

        Parameters
        ----------
        X: Ignored
            Not used, present for API consistency by convention.
        y: Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        labels: ndarray of shape (n_samples,)
            Returns +1 for samples that should be used for model training, -1 for those rejected
        """
        sorted_idxs = np.argsort(self.time_vector)
        subset_idxs = sorted_idxs[-self.nr_of_samples_to_pick :]
        self.labels[subset_idxs] = 1
        return self.labels
