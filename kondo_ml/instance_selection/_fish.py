import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import euclidean_distances, mean_squared_error
from sklearn.model_selection import cross_validate

from kondo_ml.instance_selection.base import SelectorMixin
from kondo_ml.utils import normalize_array, train_lr_model


class Fish1Selector(SelectorMixin, BaseEstimator):
    """FISH1 instance selection algorithm from the paper Combining Similarity in Time and Space for
    Training Set Formation under Concept Drift by Zliobaite
    https://eprints.bournemouth.ac.uk/18567/1/FISH_journal_preprint.pdf
    """

    def __init__(
        self,
        temporal_weight=0.5,
        subsize_frac=1,
    ):
        super().__init__(subsize_frac)
        self.temporal_weight = temporal_weight
        self.spatial_weight = 1 - temporal_weight

    def calc_time_space_dist(self, X_spatial) -> np.array:
        """Returns weighted combined spatial and temporal distance for each sample in X

        Parameters
        ----------
        X_spatial: {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        array of combined temporal and spatial distance
        """
        x_target_space = self.x_target
        if x_target_space.ndim == 1:
            x_target_space = x_target_space.reshape(1, -1)
        d_space = euclidean_distances(X_spatial, x_target_space)
        d_time = np.abs(self.X_temporal - self.x_target_temporal).astype("float32")
        d_space = normalize_array(d_space)
        d_time = normalize_array(d_time)
        return self.spatial_weight * d_space.flatten() + self.temporal_weight * d_time

    def check_valid_weights(self):
        """Raises a ValueError if the temporal weight is below zero or above one"""
        if self.temporal_weight > 1 or self.temporal_weight < 0:
            raise ValueError("invalid weight")

    def fit(self, X, y=None):
        """Fit the algorithm according to the given training data

        The time vector needs to be passed as the last column in the X array. And the reference sample needs to be
        the last row in X

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
        self.X_temporal = pd.to_datetime(X[:-1, -1]).to_numpy()
        self.x_target = X[-1, :-1]
        self.x_target_temporal = pd.to_datetime(X[-1, -1]).to_numpy()
        self.nr_samples = X.shape[0] - 1
        self.samples_to_pick = self.calc_subset_sizeint(self.nr_samples)
        # pseudo 1.
        self.distance_vector = self.calc_time_space_dist(X[:-1, :-1])
        self.scores = -1 * self.distance_vector
        self.labels = np.ones(self.nr_samples, dtype="int8") * -1
        return self

    def predict(self, X, y=None):
        """Predict the labels (1 use for training, -1 rejected) of X that was passed to the fit function

         Parameters
         ----------
        X: Ignored
             Not used, present for API consistency by convention.
         y: Ignored
             Not used, present for API consistency by convention.

         Returns
         -------

        """
        idx_to_pick = np.argsort(self.distance_vector, axis=0).flatten()[
            : self.samples_to_pick
        ]  # pick smallest dist
        self.labels[idx_to_pick] = 1
        return self.labels


class Fish2Selector(Fish1Selector):
    def __init__(
        self,
        spatial_weight,
        temporal_weight,
        neighborhood_size,
        subsize_frac=1,
    ):
        super().__init__(spatial_weight, temporal_weight, subsize_frac)
        self.k = neighborhood_size

    def predict(self, X, y=None):
        X_without_target_nor_time = X[:-1, :-1].copy()
        errors = {}
        metric = mean_squared_error
        sorted_idxs = np.argsort(self.distance_vector, axis=0).flatten()  # 2
        # In paper they use step size 5
        for N in range(self.k, self.nr_samples, 50):
            # 3b
            linreg = LinearRegression()
            cv_results = cross_validate(
                linreg,
                X_without_target_nor_time[sorted_idxs[:N], :],
                y[sorted_idxs[:N]],
                cv=N,
                scoring="neg_mean_squared_error",
            )
            leave_out_idx = np.argsort(cv_results["test_score"])[0]
            model_N = train_lr_model(
                np.delete(
                    X_without_target_nor_time[sorted_idxs[:N], :], leave_out_idx, axis=0
                ),
                np.delete(y[sorted_idxs[:N]], leave_out_idx),
            )
            # 3c
            predicted_y_k = model_N.predict(
                X_without_target_nor_time[sorted_idxs[: self.k], :]
            )
            errors[N] = metric(y[sorted_idxs[: self.k]], predicted_y_k)
        set_size_min_error = min(errors, key=errors.get)  # 4
        self.labels[sorted_idxs[:set_size_min_error]] = 1  # 5
        return self.labels
