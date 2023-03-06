import numpy as np

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, r2_score

from ..model import get_lr_model_random_params
from .base import SelectorMixin


class GradientShapleySelector(SelectorMixin, BaseEstimator):
    def __init__(
        self,
        convergence_error=0.05,
        learning_rate=0.001,
        subsize_frac=0.8,
        max_iteration=20,
    ):
        super().__init__(subsize_frac)
        self.convergence_error = convergence_error
        self.lr = learning_rate
        self.max_iteration = max_iteration

    def single_sample_gradient_descent_mse(self, model, single_x, y_true, lr):
        y_pred = model.predict(single_x)
        loss = (y_true - y_pred) ** 2
        d_intercept = 2 * (y_true - y_pred)
        d_coef = 2 * single_x * (y_pred - y_true)
        model.coef_ = model.coef_ - lr * d_coef
        model.intercept_ = model.intercept_ - lr * d_intercept
        return model, loss

    def error(self, iteration):
        """
        Parameters
        ----------
        iteration

        Returns
        -------

        """
        if iteration < 100:
            return 1.0
        nominator = np.abs(self.scores**iteration - self.scores ** (iteration - 100))
        denominator = np.abs(self.scores)
        error = (1 / self.nr_samples) * np.sum(nominator / denominator)
        return error

    def fit(self, X, y=None):
        self.nr_samples = X.shape[0]
        self.nr_features = X.shape[1]
        self.scores = np.zeros(self.nr_samples)
        self.subsize_int = self.calc_subset_sizeint(self.nr_samples)
        self.labels = np.ones(self.nr_samples, dtype="int8") * -1
        return self

    def predict(self, X, y):
        metric = r2_score
        performance_scores = np.zeros(self.nr_samples)
        t = 0
        permutated_idxs_last_iteration = np.arange(
            self.nr_samples
        )  # just so that 1st iteration works (values will be 0)
        while self.error(t) >= self.convergence_error and t < self.max_iteration:
            t += 1
            if t % 5 == 0:
                print("iteration: ", t)
            permutated_idxs = np.random.permutation(self.nr_samples)
            model = get_lr_model_random_params(self.nr_features, model_type="linear")
            y_pred_rnd = model.predict(X)
            # Because it is a score we need to convert errors into a score where bigger = better
            performance_scores[0] = metric(y_true=y, y_pred=y_pred_rnd)
            for j in range(self.nr_samples):
                model, single_loss = self.single_sample_gradient_descent_mse(
                    model,
                    X[permutated_idxs[j], :].reshape(1, -1),
                    y[permutated_idxs[j]],
                    self.lr,
                )
                performance_scores[j] = metric(y, model.predict(X))
                self.scores[permutated_idxs[j]] = (
                    ((t - 1) / t) * self.scores[permutated_idxs_last_iteration[j]]
                ) + ((1 / t) * (performance_scores[j] - performance_scores[j - 1]))
            permutated_idxs_last_iteration = permutated_idxs.copy()
        sorted_idx = np.flip(np.argsort(self.scores))
        self.labels[sorted_idx[: self.subsize_int]] = 1
        return self.labels
