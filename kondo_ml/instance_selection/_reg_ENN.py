import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors

from kondo_ml.instance_selection.base import SelectorMixin
from kondo_ml.utils import train_lr_model, weighted_avg_and_std


class RegEnnSelector(SelectorMixin, BaseEstimator):
    def __init__(self, alpha: float = 5, nr_of_neighbors: int = 9, subsize_frac=1):
        super().__init__(subsize_frac=subsize_frac)
        self.nr_of_neighbors = nr_of_neighbors
        self.alpha = alpha

    def fit(self, X, y):
        """
        Fit the algorithm according to the given training data.

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
        self.nr_of_samples = X.shape[0]
        self.labels = np.ones(self.nr_of_samples, dtype="int8") * -1
        self.scores = np.ones(self.nr_of_samples, dtype="float32")
        return self

    def predict_instance_from_model_without_that_instance(
        self,
        X,
        y,
        i,
        subset_mask,
    ):
        """

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : array-like of shape (n_samples)
            True target values for X
        i : int
            Index of the instance to exclude
        subset_mask

        Returns
        -------

        """
        investigated_instance = X[i, :].reshape(1, -1)
        mask_model_training = subset_mask.copy()
        mask_model_training[
            i
        ] = False  # remove investigated instance from model training
        model = train_lr_model(X[mask_model_training, :], y[mask_model_training])  # 2

        y_pred = model.predict(investigated_instance)  # 2
        return investigated_instance, y_pred

    def get_neighbors_indices(self, X, investigated_instance, subset_mask):
        nbrs = NearestNeighbors(
            n_neighbors=self.nr_of_neighbors + 1, algorithm="auto"
        ).fit(X[subset_mask, :])
        indices = nbrs.kneighbors(investigated_instance, return_distance=False)
        indices = indices[0, 1:]  # 3
        return indices

    def predict(self, X, y):
        """Predict the labels (1 use for training, -1 rejected) of X according to RegENN

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        labels: ndarray of shape (n_samples,)
            Returns +1 for samples that should be used for model training, -1 for those rejected
        """
        subset_mask = np.ones(self.nr_of_samples, dtype="bool")
        for i in range(self.nr_of_samples):
            (
                investigated_instance,
                y_pred,
            ) = self.predict_instance_from_model_without_that_instance(
                X, y, i, subset_mask
            )
            # As the closest neighbor is always the instance itself, we add one neighbor and ignore the 0 index
            indices = self.get_neighbors_indices(X, investigated_instance, subset_mask)
            theta = self.alpha * np.std(y[indices])  # 4
            y_true = y[i]
            self.scores[i] = (np.abs(y_true - y_pred) - theta) * -1
            if np.abs(y_true - y_pred) > theta:  # 5
                subset_mask[i] = False  # 6
            if sum(subset_mask) <= self.nr_of_neighbors + 1:
                print("not converged - all samples got kicked out")
                break
        self.labels[subset_mask] = 1  # 7
        return self.labels


class RegENNSelectorTime(RegEnnSelector):
    """RegENNTime is an adaption of RegENN, presented by the author of this package.

    RegENNTime makes use of a time vector by calculating weights according to the distance of each instance
    to the last (newest) instance in the set. These weights are used to calculate a weighted
    standard deviation which is the basis for the threshold theta and the selection criterion.
    """

    def __init__(
        self,
        alpha=5,
        nr_of_neighbors=9,
        subsize_frac=1,
        time_scaling_factor=300,
        distance_measure: str = "linear",
    ):
        super().__init__(alpha, nr_of_neighbors, subsize_frac)
        self.time_scaling_factor = time_scaling_factor
        self.distance_measure = distance_measure

    def fit(self, X, y):
        """
        Gets the time information from the input variable. It is important that that the time feature is the last column
        of the X array.

        Parameters
        ----------
        X
            Input array with (n_samples x n_features), where the last column contains the time of each instance
        y
            True target values for X
        Returns
        -------

        """
        self.nr_of_samples = X.shape[0]
        self.labels = np.ones(self.nr_of_samples, dtype="int8") * -1
        self.scores = np.ones(self.nr_of_samples, dtype="float32")
        self.time_vector = pd.to_datetime(X[:, -1]).to_numpy()
        self.reference_time = self.time_vector[-1] + np.timedelta64(1, "D")
        return self

    def get_weight_by_time(self, time_vector_neighbors, time_delta_in="D"):
        if self.distance_measure == "linear":
            time_distances = (
                np.abs(time_vector_neighbors - self.reference_time)
                .astype(f"timedelta64[{time_delta_in}]")
                .astype(float)  # this makes it the number of days
            )
        elif self.distance_measure == "exp":
            time_distances = np.exp(
                np.abs(time_vector_neighbors - self.reference_time)
                .astype(f"timedelta64[{time_delta_in}]")
                .astype(float)
            )
        time_similarity = self.time_scaling_factor / (time_distances + 1.000000e-10)
        weights = softmax(time_similarity)
        return weights.flatten()

    def predict(self, X, y):
        """Predict the labels (1 use for training, -1 rejected) of X according to RegENNTime


        As the time vector is included in the fit method as the last column of input array X,
        it is also expected here and the last column therefore removed from
        the calculations.

        Parameters
        ----------
        X
            Input array with (n_samples x n_features), where the last column contains the time of each instance
        y
            Ground-truth target values for X

        Returns
        -------
        labels: ndarray of shape (n_samples,)
            Returns +1 for samples that should be used for model training, -1 for those rejected
        """
        X_without_time = X[:, :-1].copy()
        subset_mask = np.ones(self.nr_of_samples, dtype="bool")
        for i in range(self.nr_of_samples):
            (
                investigated_instance,
                y_pred,
            ) = self.predict_instance_from_model_without_that_instance(
                X_without_time, y, i, subset_mask
            )
            # As the closest neighbor is always the instance itself, we add one neighbor and ignore the 0 index
            indices = self.get_neighbors_indices(
                X_without_time, investigated_instance, subset_mask
            )
            weights = self.get_weight_by_time(self.time_vector[indices])
            _, weighted_std = weighted_avg_and_std(y[indices], weights)
            theta = self.alpha * weighted_std  # 4
            y_true = y[i]
            self.scores[i] = (np.abs(y_true - y_pred) - theta) * -1
            if np.abs(y_true - y_pred) > theta:  # 5
                subset_mask[i] = False  # 6
            if sum(subset_mask) <= self.nr_of_neighbors + 1:
                print("not converged - all samples got kicked out")
                break
        self.labels[subset_mask] = 1  # 7
        return self.labels
