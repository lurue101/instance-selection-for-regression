import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

import kondo_ml
from kondo_ml.instance_selection.base import SelectorMixin
from kondo_ml.utils import train_lr_model, transform_selector_output_into_mask


class DROPRESuperClass(BaseEstimator, SelectorMixin):
    """
    Class that contain the basic function that DROP variants 2/3 - RE/RT share.
    The algorithm can be found in the paper by Arnaiz-Gonzalez et al.
    https://www.sciencedirect.com/science/article/abs/pii/S0925231216301953
    """

    def __init__(self, nr_of_neighbors: int = 5, subsize_frac=1):
        super().__init__(subsize_frac=subsize_frac)
        self.k = nr_of_neighbors

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
        self.scores = np.zeros(self.nr_of_samples, dtype="float32")
        return self

    def predict(self, X, y):
        """Each Sub-DROP algorithm has to implement their own predict method"""
        raise ValueError("implement in subclass")

    def find_neighbors_and_associates(self, X: np.ndarray, invalid_indices: list):
        """
        Creates a dictionary that contain the nearest neighbors and associates for each sample. The dictionary keys are
        the indices in the order of X.
        A sample "i" is an associate of sample "j" if "j" is in the list of nearest neighbors of "i"
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        invalid_indices
            indices that should not be included (because they've been filtered before)
        Returns
        -------
        two dicts - one for the neighbors , one for the associates
        """
        nbrs = NearestNeighbors(n_neighbors=self.k + 25, algorithm="auto").fit(X)
        dict_neighbors = {}
        dict_associates = {idx: [] for idx in range(self.nr_of_samples)}
        for i in range(self.nr_of_samples):
            if i in invalid_indices:
                continue
            investigated_instance = X[i, :].reshape(1, -1)
            indices = nbrs.kneighbors(investigated_instance, return_distance=False)
            # skip first zero index, as it is the sample itself
            indices = indices[0, 1:]
            # filter for samples that were removed in an earlier step, as they shouldn't be part of any consideration
            indices = np.array([idx for idx in indices if idx not in invalid_indices])
            dict_neighbors[i] = indices
            # Add instance i as associate for all it's k nearest neighbors
            for neighbor in indices[: self.k]:
                dict_associates[neighbor].append(i)
        return dict_neighbors, dict_associates

    def prepare_subset_mask(self, X, y) -> np.ndarray:
        """
        This function is used to apply a instance selection before the main DROP algorithm is run. In the basic
        version here it just returns boolean indexes that choose every instance

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y
            array containing target variable
        Returns
        -------
        array containing a boolean mask indicating which instances to further use
        """
        return np.ones(self.nr_of_samples, dtype="bool")


class DROPRE2RE(DROPRESuperClass):
    """DROP2 algorithm for Regression using the cumulative Error approach"""

    def main_re_loop(
        self, X, y, subset_mask, loop_idx, dict_neighbors, dict_associates
    ) -> np.ndarray:
        """
        This is the main function of those algorithms that use cumulative error as a criteria to decide which instances
        to include

        The numbers in the comments refer to the pseudo code lines from the DROP for regression paper (Arnaiz, 2016)

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y
            array containing target variable
        subset_mask
            boolean mask to indicate which instances to inspect
        loop_idx
            list of indexes to loop through
        dict_neighbors
            dict containing the neighbors for each instance (via index)
        dict_associates
            dict containing the associates for each instance (via index)
        Returns
        -------
        boolean array, True indicating that the instance should be chosen for model training
        """
        for i in loop_idx:  # 5
            error_with = 0  # 6
            error_without = 0  # 7
            for a_idx in dict_associates[i]:  # 8
                a_all_neighbors = dict_neighbors[a_idx]
                a_nn_with = a_all_neighbors[: self.k]
                model_with = train_lr_model(X[a_nn_with, :], y[a_nn_with])
                error_with += np.abs(
                    y[a_idx] - model_with.predict(X[a_idx, :].reshape(1, -1))
                )  # 9
                a_nn_without = np.delete(a_all_neighbors, a_all_neighbors == i)[
                    : self.k
                ]
                model_without = train_lr_model(X[a_nn_without, :], y[a_nn_without])
                error_without += np.abs(
                    y[a_idx] - model_without.predict(X[a_idx, :].reshape(1, -1))
                )  # 10
            self.scores[i] = (error_with - error_without) * -1
            if error_without <= error_with:  # 11
                subset_mask[i] = False  # 12
                if sum(subset_mask) < self.k:
                    print("basically all samples kicked out")
                    return subset_mask
                for a_idx in dict_associates[i]:  # 13
                    neighbors = dict_neighbors[a_idx]
                    dict_neighbors[a_idx] = np.delete(neighbors, neighbors == i)  # 15
                    # 16 is covered above by calculating more neighbors than needed
                    if len(dict_neighbors[a_idx]) < self.k:
                        # implement to add more neighbors
                        raise ValueError("not enough neighbors")
        return subset_mask

    def predict(self, X, y):
        """Predict the labels (1 use for training, -1 rejected) of X according to DROP

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

        subset_mask = self.prepare_subset_mask(X, y)
        dict_neighbors, dict_associates = self.find_neighbors_and_associates(
            X, []
        )  # 2,3,4
        subset_mask = self.main_re_loop(
            X,
            y,
            subset_mask,
            range(self.nr_of_samples),
            dict_neighbors,
            dict_associates,
        )
        self.labels[subset_mask] = 1  # 17
        return self.labels


class DROP2RT(DROPRE2RE):
    """DROP2 algorithm for Regression using the adaptive Thresholding approach"""

    def __init__(self, alpha: float = 0.5, nr_of_neighbors: int = 9, subsize_frac=1):
        super().__init__(
            nr_of_neighbors,
            subsize_frac,
        )
        self.alpha = alpha

    def main_rt_loop(
        self, X, y, sorted_idx, dict_neighbors, dict_associates, subset_mask
    ):
        """
        This is the main function of those algorithms
        that use adaptive thresholding as a criteria to decide which instances to include

        The numbers in the comments refer to the pseudo code lines from the DROP for regression paper (Arnaiz, 2016)

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y
            array containing target variable
        sorted_idx
            list of indexes to loop through
        dict_neighbors
            dict containing the neighbors for each instance (via index)
        dict_associates
            dict containing the associates for each instance (via index)

        Returns
        -------
        boolean array, True indicating that the instance should be chosen for model training
        """

        for i in sorted_idx:  # 6
            threshold_with = 0
            threshold_without = 0
            for a_idx in dict_associates[i]:
                a_all_neighbors = dict_neighbors[a_idx]
                a_nn_with = a_all_neighbors[: self.k]
                theta = self.get_theta(y[a_nn_with])
                model_with = train_lr_model(X[a_nn_with, :], y[a_nn_with])
                if (
                    np.abs(y[a_idx] - model_with.predict(X[a_idx, :].reshape(1, -1)))
                    <= theta
                ):
                    threshold_with += 1

                a_nn_without = np.delete(a_all_neighbors, a_all_neighbors == i)[
                    : self.k
                ]
                model_without = train_lr_model(X[a_nn_without, :], y[a_nn_without])
                if (
                    np.abs(y[a_idx] - model_without.predict(X[a_idx, :].reshape(1, -1)))
                    <= theta
                ):
                    threshold_without += 1
            self.scores[i] = (threshold_without - threshold_with) * -1
            if threshold_without >= threshold_with:
                subset_mask[i] = False  # 16
                if sum(subset_mask) < self.k:
                    print("basically all samples kicked out")
                    return subset_mask
                for a_idx in dict_associates[i]:
                    neighbors = dict_neighbors[a_idx]
                    dict_neighbors[a_idx] = np.delete(neighbors, neighbors == i)  # 18
                    if len(dict_neighbors[a_idx]) < self.k:
                        # implement to add more neighbors
                        return subset_mask
        return subset_mask

    def predict(self, X, y):
        """Predict the labels (1 use for training, -1 rejected) of X according to DROP

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
        subset_mask = self.prepare_subset_mask(X, y)
        invalid_indices = np.argwhere(subset_mask == False).flatten()
        # sort idx, as the order in which the loop runs makes a difference
        sorted_idx = self.get_sorted_idx_by_dist_to_closest_enemy(X, y)  # 2
        if sum(subset_mask) < self.k + 25:
            print("No instances left after RegENN")
            self.labels[subset_mask] = 1
            self.scores[invalid_indices] = -1
            return self.labels
        dict_neighbors, dict_associates = self.find_neighbors_and_associates(
            X, invalid_indices
        )  # 3,4,5
        # kick out idx that were removed by the noise filter
        loop_idx = np.array([idx for idx in sorted_idx if idx not in invalid_indices])
        subset_mask = self.main_rt_loop(
            X, y, loop_idx, dict_neighbors, dict_associates, subset_mask
        )
        self.labels[subset_mask] = 1
        self.scores[invalid_indices] = -1
        return self.labels

    def get_sorted_idx_by_dist_to_closest_enemy(self, X, y):
        """Returns a sorted list of indices. Sorted ascendingly by the distance to the closest enemy.

        An enemy of a sample x_i is a sample x_j, where |Y(x_i) - Y(x_j)| > theta

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y
            array containing target variable

        Returns
        -------
        sorted_idx: ndarray of shape (n_samples,)
            indices of the samples in X, sorted ascendingly by distance to their closest enemy
        """
        # enemy is closest sample x_j where |Y(x_i) - Y(x_j)| > theta
        model = train_lr_model(X, y)
        closest_enemy_distances = self.find_dist_to_closest_enemy_for_all_instances(
            X, y, model
        )
        sorted_idx = np.argsort(closest_enemy_distances)
        return sorted_idx

    def find_dist_to_closest_enemy_for_all_instances(self, X, y, model):
        """

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y
            array containing target variable
        model
            fitted sklearn regression model

        Returns
        -------
        closest_enemy_distances: ndarray of shape (n_samples,)
            each index contains the distance to the closest enemy
            for the samples corresponding to the same index in X
        """
        closest_enemy_distances = np.ones(X.shape[0])
        theta = self.get_theta(y)
        predictions = model.predict(X)
        for i in range(X.shape[0]):
            investigated_instance = X[i, :].reshape(1, -1)
            abs_diff_predictions = np.abs(
                predictions - model.predict(investigated_instance)
            )
            enemies_idx = np.argwhere(abs_diff_predictions > theta).flatten()
            if len(enemies_idx) == 0:
                closest_enemy_distances[i] = abs_diff_predictions[
                    abs_diff_predictions > 0
                ].min()
            else:
                distances_enemies = pairwise_distances(
                    X[enemies_idx, :], investigated_instance
                )
                lowest_dist_to_enemy = np.min(distances_enemies)
                closest_enemy_distances[i] = lowest_dist_to_enemy
        return closest_enemy_distances

    def get_theta(self, y_sub):
        """Returns the adaptive threshold :math:`\theta`

        Parameters
        ----------
        y_sub
            subset of the target vector y

        Returns
        -------
        theta: np.float
        """
        theta = self.alpha * np.std(y_sub)
        return theta


class DROP3RE(DROP2RT):
    """DROP3 algorithm for Regression using the cumulative Error approach"""

    def __init__(
        self,
        alpha: float = 0.5,
        nr_of_neighbors: int = 9,
        subsize_frac=1,
        reg_enn_alpha=5,
        reg_enn_neighbors=9,
    ):
        super().__init__(
            alpha=alpha,
            nr_of_neighbors=nr_of_neighbors,
            subsize_frac=subsize_frac,
        )
        self.reg_enn_alpha = reg_enn_alpha
        self.reg_enn_neighbors = reg_enn_neighbors

    def predict(self, X, y):
        """Predict the labels (1 use for training, -1 rejected) of X according to DROP

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
        subset_mask = self.prepare_subset_mask(X, y)
        invalid_indices = np.argwhere(subset_mask == False).flatten()
        # sort idx, as the order in which the loop runs makes a difference
        sorted_idx = self.get_sorted_idx_by_dist_to_closest_enemy(X, y)  # 2
        dict_neighbors, dict_associates = self.find_neighbors_and_associates(
            X, invalid_indices
        )  # 3,4,5
        # kick out idx that were removed by the noise filter
        loop_idx = np.array([idx for idx in sorted_idx if idx not in invalid_indices])
        subset_mask = self.main_re_loop(
            X, y, subset_mask, loop_idx, dict_neighbors, dict_associates
        )
        self.labels[subset_mask] = 1  # 17
        self.scores[invalid_indices] = -1
        return self.labels

    def noise_filter(self, X, y):
        """Predict the labels (1 clean samples, -1 noisy samples) of X according to RegENN

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
            Returns +1 for "clean" samples, -1 for noisy samples
        """
        regenn = kondo_ml.instance_selection.RegEnnSelector(
            alpha=self.reg_enn_alpha,
            nr_of_neighbors=self.reg_enn_neighbors,
            subsize_frac=self.subsize_frac,
        )
        labels = regenn.fit_predict(X, y)
        return transform_selector_output_into_mask(labels)

    def prepare_subset_mask(self, X, y):
        """Returns pre-selection of samples to use for DROP algorithm selection

        Parameters
        ----------
         X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        subset_mask
            True for samples to inspect in DROP algorith, False for rejected samples
        """
        subset_mask = self.noise_filter(X, y)
        return subset_mask


class DROP3RT(DROP2RT):
    """DROP3 algorithm for Regression using adaptive Thresholding"""

    def __init__(
        self,
        alpha: float = 0.5,
        nr_of_neighbors: int = 9,
        subsize_frac=1,
        reg_enn_alpha=5,
        reg_enn_neighbors=9,
    ):
        super().__init__(
            alpha=alpha,
            nr_of_neighbors=nr_of_neighbors,
            subsize_frac=subsize_frac,
        )
        self.reg_enn_alpha = reg_enn_alpha
        self.reg_enn_neighbors = reg_enn_neighbors

    def noise_filter(self, X, y):
        """Predict the labels (1 clean samples, -1 noisy samples) of X according to RegENN

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
            Returns +1 for "clean" samples, -1 for noisy samples
        """
        regenn = kondo_ml.instance_selection.RegEnnSelector(
            alpha=self.reg_enn_alpha,
            nr_of_neighbors=self.reg_enn_neighbors,
            subsize_frac=self.subsize_frac,
        )
        labels = regenn.fit_predict(X, y)
        return transform_selector_output_into_mask(labels)

    def prepare_subset_mask(self, X, y):
        """Returns pre-selection of samples to use for DROP algorithm selection

        Parameters
        ----------
         X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        subset_mask
            True for samples to inspect in DROP algorith, False for rejected samples
        """
        subset_mask = self.noise_filter(X, y)
        return subset_mask
