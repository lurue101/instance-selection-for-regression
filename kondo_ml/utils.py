import numpy as np
from sklearn.linear_model import LassoLarsCV, LinearRegression


def convert_recorded_at_to_seconds(datetime_values):
    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    return (datetime_values - unix_epoch) / one_second


def transform_selector_output_into_mask(labels):
    """
    Parameters
    ----------
    labels

    Returns
    -------

    """
    mask_labels = labels.copy()
    mask_labels[mask_labels == -1] = 0
    return mask_labels.astype(bool)


def normalize_array(array):
    min_value = np.min(array)
    max_value = np.max(array)
    abs_max_diff = np.abs(max_value - min_value)
    return (array - min_value) / abs_max_diff


def normalize_array_neg_plus_1(array):
    min_value = np.min(array)
    max_value = np.max(array)
    abs_max_diff = np.abs(max_value - min_value)
    return (2 * (array - min_value) / abs_max_diff) - 1


def calc_pct_increase(og_number, new_number):
    increase = new_number - og_number
    pct_increase = 100 * increase / og_number
    return pct_increase


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return (average, np.sqrt(variance))


def train_lr_model(X, y, model_type: str = "linear"):
    if model_type == "linear":
        reg = LinearRegression()
    elif model_type == "lasso_lars_cv":
        reg = LassoLarsCV(normalize=False, max_iter=500)
    else:
        raise ValueError(f"{model_type} is not implemented")
    reg.fit(X, y)
    return reg


def get_random_intercept_coef(nr_of_features):
    rnd_intercept = np.random.normal(0, 2)
    rnd_coefs = np.random.normal(0, 2, nr_of_features)
    return (rnd_intercept, rnd_coefs)


def get_lr_model_random_params(nr_of_features, model_type="linear"):
    if model_type == "linear":
        reg = LinearRegression()
    elif model_type == "lasso_lars_cv":
        reg = LassoLarsCV(normalize=False, max_iter=50)
    else:
        raise ValueError("we don't have this model")
    rnd_intercept, rnd_coefs = get_random_intercept_coef(nr_of_features)
    reg.intercept_ = rnd_intercept
    reg.coef_ = rnd_coefs
    return reg
