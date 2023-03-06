class SelectorMixin:
    _estimator_type = "selector"

    def __init__(self, subsize_frac):
        self.subsize_frac = subsize_frac

    def fit_predict(self, X, y=None):
        """Perform fit on X and returns labels for X.

        Returns -1 for outliers and 1 for inliers.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            1 for inliers, -1 for outliers.
        """
        # override for transductive outlier detectors like LocalOulierFactor
        return self.fit(X, y).predict(X, y)

    def calc_subset_sizeint(self, nr_samples):
        return int(nr_samples * self.subsize_frac)
