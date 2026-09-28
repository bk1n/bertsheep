import numpy as np
from xgboost import XGBRegressor

# The settings the search leaves alone. Early stopping on valid picks the number
# of trees, so n_estimators is only a ceiling: set high enough that early
# stopping, not the cap, ends a fit at the lowest learning rate searched.
XGB_PARAMS = {
    "n_estimators": 5000,
    "early_stopping_rounds": 50,
}


def fit_baseline(fingerprints: np.ndarray, labels: np.ndarray, train: np.ndarray,
                 valid: np.ndarray, seed: int,
                 params: dict[str, float | int]) -> XGBRegressor:
    """
    Fit XGBoost on ECFP4 bits: what a model with no language pretraining gets
    from the same split. Early stopping watches valid, as the transformer arms'
    epoch selection does, so the search and the matrix fit the baseline the
    same way and test is never read here.

    Parameters
    ----------
    fingerprints : np.ndarray
        (n, bits) fingerprint matrix over the whole frame.
    labels : np.ndarray
        Labels over the whole frame, in the same order.
    train : np.ndarray
        Positional indices to fit on.
    valid : np.ndarray
        Positional indices early stopping watches.
    seed : int
        Seed for XGBoost's row and column subsampling.
    params : dict[str, float | int]
        Searched hyperparameters, on top of XGB_PARAMS.

    Returns
    -------
    XGBRegressor
        The fitted model; with early stopping set, predict() uses the best
        round, not the last.
    """
    model = XGBRegressor(**XGB_PARAMS, **params, random_state=seed)
    model.fit(fingerprints[train], labels[train],
              eval_set=[(fingerprints[valid], labels[valid])], verbose=False)
    return model
