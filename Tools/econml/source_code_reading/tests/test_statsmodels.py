# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

import numpy as np
import pytest

import scipy.special
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.regression.linear_model import WLS
from statsmodels.sandbox.regression.gmm import IV2SLS
from statsmodels.tools.tools import add_constant

from econml.inference import StatsModelsInference, StatsModelsInferenceDiscrete
from econml.dml import LinearDML, NonParamDML
from econml.dr import LinearDRLearner
from econml.iv.dml import DMLIV
from econml.iv.dr import LinearDRIV
from econml.sklearn_extensions.linear_model import WeightedLasso, StatsModelsLinearRegression
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression as OLS
from econml.sklearn_extensions.linear_model import StatsModels2SLS
from econml.utilities import (ndim, transpose, shape, reshape, hstack, WeightedModelWrapper)


class StatsModelsOLS:
    """
    Helper class to wrap a StatsModels OLS model to conform to the sklearn API.

    Parameters
    ----------
    fit_intercept: bool (default False)
        Whether to fit an intercept

    Attributes
    ----------
    fit_args: dict of str: object
        The arguments to pass to the `OLS` regression's `fit` method.  See the
        statsmodels documentation for more information.
    results: RegressionResults
        After `fit` has been called, this attribute will store the regression results.
    """

    def __init__(self, fit_intercept=True, fit_args={}):
        self.fit_args = fit_args
        self.fit_intercept = fit_intercept

    def fit(self, X, y, sample_weight=None):
        """
        Fit the ordinary least squares model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data
        y : array_like, shape (n_samples, 1) or (n_samples,)
            Target values
        sample_weight : array_like, shape (n_samples,)
            Individual weights for each sample

        Returns
        -------
        self
        """
        assert ndim(y) == 1 or (ndim(y) == 2 and shape(y)[1] == 1)
        y = reshape(y, (-1,))
        if self.fit_intercept:
            X = add_constant(X, has_constant='add')
        if sample_weight is not None:
            ols = WLS(y, X, weights=sample_weight, hasconst=self.fit_intercept)
        else:
            ols = WLS(y, X, hasconst=self.fit_intercept)
        self.results = ols.fit(**self.fit_args)
        return self

    def predict(self, X):
        """
        Predict using the model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        array, shape (n_samples,)
            Predicted values
        """
        if self.fit_intercept:
            X = add_constant(X, has_constant='add')
        return self.results.predict(X)

    def predict_interval(self, X, alpha=.05):
        """
        Get a confidence interval for the prediction at `X`.

        Parameters
        ----------
        X : array_like
            The features at which to predict
        alpha : float
            The significance level to use for the interval

        Returns
        -------
        array, shape (2, n_samples)
            Lower and upper bounds for the confidence interval at each sample point
        """
        if self.fit_intercept:
            X = add_constant(X, has_constant='add')
        # NOTE: we use `obs = False` to get a confidence, rather than prediction, interval
        preds = self.results.get_prediction(X).conf_int(alpha=alpha, obs=False)
        # statsmodels uses the last dimension instead of the first to store the confidence intervals,
        # so we need to transpose the result
        return transpose(preds)

    @property
    def coef_(self):
        if self.fit_intercept:
            return self.results.params[1:]
        else:
            return self.results.params

    def coef__interval(self, alpha):
        if self.fit_intercept:
            return transpose(self.results.conf_int(alpha=alpha)[1:])
        else:
            return transpose(self.results.conf_int(alpha=alpha))

    @property
    def intercept_(self):
        if self.fit_intercept:
            return self.results.params[0]
        else:
            return 0

    def intercept__interval(self, alpha):
        if self.fit_intercept:
            return self.results.conf_int(alpha=alpha)[0]
        else:
            return np.array([0, 0])


def _compare_classes(est, lr, X_test, alpha=.05, tol=1e-12):
    assert np.all(np.abs(est.coef_ - lr.coef_) < tol), "{}, {}".format(est.coef_, lr.coef_)
    assert np.all(np.abs(np.array(est.coef__interval(alpha=alpha)) -
                         np.array(lr.coef__interval(alpha=alpha))) < tol),\
        "{}, {}".format(est.coef__interval(alpha=alpha), np.array(lr.coef__interval(alpha=alpha)))
    assert np.all(np.abs(est.intercept_ - lr.intercept_) < tol), "{}, {}".format(est.intercept_, lr.intercept_)
    assert np.all(np.abs(np.array(est.intercept__interval(alpha=alpha)) -
                         np.array(lr.intercept__interval(alpha=alpha))) < tol),\
        "{}, {}".format(est.intercept__interval(alpha=alpha), lr.intercept__interval(alpha=alpha))
    assert np.all(np.abs(est.predict(X_test) - lr.predict(X_test)) <
                  tol), "{}, {}".format(est.predict(X_test), lr.predict(X_test))
    assert np.all(np.abs(np.array(est.predict_interval(X_test, alpha=alpha)) -
                         np.array(lr.predict_interval(X_test, alpha=alpha))) < tol),\
        "{}, {}".format(est.predict_interval(X_test, alpha=alpha), lr.predict_interval(X_test, alpha=alpha))


def _summarize(X, y, w=None):
    original_w = w
    if w is None:
        w = np.ones((y.shape[0],))
    X_unique = np.unique(X, axis=0)
    y_sum_first = []
    n_sum_first = []
    var_first = []
    X_final_first = []
    w_sum_first = []
    y_sum_sec = []
    n_sum_sec = []
    var_sec = []
    X_final_sec = []
    w_sum_sec = []
    X1 = []
    X2 = []
    y1 = []
    y2 = []
    w1 = []
    w2 = []
    for it, xt in enumerate(X_unique):
        mask = (X == xt).all(axis=1)
        if mask.any():
            y_mask = y[mask]
            X_mask = X[mask]
            w_mask = w[mask]
            if np.sum(mask) >= 2:
                X_mask_first = X_mask[:y_mask.shape[0] // 2]
                X_mask_sec = X_mask[y_mask.shape[0] // 2:]
                y_mask_first = y_mask[:y_mask.shape[0] // 2]
                y_mask_sec = y_mask[y_mask.shape[0] // 2:]
                w_mask_first = w_mask[:w_mask.shape[0] // 2]
                w_mask_sec = w_mask[w_mask.shape[0] // 2:]

                X1 = np.vstack([X1, X_mask_first]) if len(X1) > 0 else X_mask_first
                y1 = np.concatenate((y1, y_mask_first)) if len(y1) > 0 else y_mask_first
                w1 = np.concatenate((w1, w_mask_first)) if len(w1) > 0 else w_mask_first
                X2 = np.vstack([X2, X_mask_sec]) if len(X2) > 0 else X_mask_sec
                y2 = np.concatenate((y2, y_mask_sec)) if len(y2) > 0 else y_mask_sec
                w2 = np.concatenate((w2, w_mask_sec)) if len(w2) > 0 else w_mask_sec

                y_sum_first.append(np.mean(y_mask_first, axis=0))
                w_sum_first.append(w_mask_first[0])
                n_sum_first.append(len(y_mask_first))
                var_first.append(np.var(y_mask_first, axis=0))
                X_final_first.append(xt)
                y_sum_sec.append(np.mean(y_mask_sec, axis=0))
                w_sum_sec.append(w_mask_sec[0])
                n_sum_sec.append(len(y_mask_sec))
                var_sec.append(np.var(y_mask_sec, axis=0))
                X_final_sec.append(xt)
            else:
                if np.random.binomial(1, .5, size=1) == 1:
                    X1 = np.vstack([X1, X_mask]) if len(X1) > 0 else X_mask
                    y1 = np.concatenate((y1, y_mask)) if len(y1) > 0 else y_mask
                    w1 = np.concatenate((w1, w_mask)) if len(w1) > 0 else w_mask
                    y_sum_first.append(np.mean(y_mask, axis=0))
                    w_sum_first.append(w_mask[0])
                    n_sum_first.append(len(y_mask))
                    var_first.append(np.var(y_mask, axis=0))
                    X_final_first.append(xt)
                else:
                    X2 = np.vstack([X2, X_mask]) if len(X2) > 0 else X_mask
                    y2 = np.concatenate((y2, y_mask)) if len(y2) > 0 else y_mask
                    w2 = np.concatenate((w2, w_mask)) if len(w2) > 0 else w_mask
                    y_sum_sec.append(np.mean(y_mask, axis=0))
                    w_sum_sec.append(w_mask[0])
                    n_sum_sec.append(len(y_mask))
                    var_sec.append(np.var(y_mask, axis=0))
                    X_final_sec.append(xt)
    if original_w is None:
        return (X1, X2, y1, y2,
                X_final_first, X_final_sec, y_sum_first, y_sum_sec, n_sum_first, n_sum_sec,
                var_first, var_sec)

    return (X1, X2, y1, y2, w1, w2,
            X_final_first, X_final_sec, y_sum_first, y_sum_sec, w_sum_first, w_sum_sec, n_sum_first, n_sum_sec,
            var_first, var_sec)


def _compare_dml_classes(est, lr, X_test, alpha=.05, tol=1e-10):
    assert np.all(np.abs(est.coef_ - lr.coef_) < tol), "{}, {}".format(est.coef_, lr.coef_)
    assert np.all(np.abs(np.array(est.coef__interval(alpha=alpha)) - np.array(lr.coef__interval(alpha=alpha))) < tol),\
        "{}, {}".format(np.array(est.coef__interval(alpha=alpha)), np.array(lr.coef__interval(alpha=alpha)))
    assert np.all(np.abs(est.effect(X_test) - lr.effect(X_test)) <
                  tol), "{}, {}".format(est.effect(X_test), lr.effect(X_test))
    assert np.all(np.abs(np.array(est.effect_interval(X_test, alpha=alpha)) -
                         np.array(lr.effect_interval(X_test, alpha=alpha))) < tol),\
        "{}, {}".format(est.effect_interval(X_test, alpha=alpha), lr.effect_interval(X_test, alpha=alpha))


def _compare_dr_classes(est, lr, X_test, alpha=.05, tol=1e-10):
    assert np.all(np.abs(est.coef_(T=1) - lr.coef_(T=1)) < tol), "{}, {}".format(est.coef_(T=1), lr.coef_(T=1))
    assert np.all(np.abs(np.array(est.coef__interval(T=1, alpha=alpha)) -
                         np.array(lr.coef__interval(T=1, alpha=alpha))) < tol),\
        "{}, {}".format(np.array(est.coef__interval(T=1, alpha=alpha)), np.array(lr.coef__interval(T=1, alpha=alpha)))
    assert np.all(np.abs(est.effect(X_test) - lr.effect(X_test)) <
                  tol), "{}, {}".format(est.effect(X_test), lr.effect(X_test))
    assert np.all(np.abs(np.array(est.effect_interval(X_test, alpha=alpha)) -
                         np.array(lr.effect_interval(X_test, alpha=alpha))) < tol),\
        "{}, {}".format(est.effect_interval(X_test, alpha=alpha), lr.effect_interval(X_test, alpha=alpha))


@pytest.mark.serial
class TestStatsModels(unittest.TestCase):

    def test_comp_with_lr(self):
        """ Testing that we recover the same as sklearn's linear regression in terms of point estimates """
        np.random.seed(123)
        n = 1000
        d = 3
        X = np.random.binomial(1, .8, size=(n, d))
        T = np.random.binomial(1, .5 * X[:, 0] + .25, size=(n,))

        def true_effect(x):
            return x[:, 0] + .5
        y = true_effect(X) * T + X[:, 0] + X[:, 2]
        weights = np.random.uniform(0, 1, size=(X.shape[0]))

        est = OLS().fit(X, y)
        lr = LinearRegression().fit(X, y)
        assert np.all(np.abs(est.coef_ - lr.coef_) < 1e-12), "{}, {}".format(est.coef_, lr.coef_)
        assert np.all(np.abs(est.intercept_ - lr.intercept_) < 1e-12), "{}, {}".format(est.coef_, lr.intercept_)

        est = OLS(fit_intercept=False).fit(X, y)
        lr = LinearRegression(fit_intercept=False).fit(X, y)
        assert np.all(np.abs(est.coef_ - lr.coef_) < 1e-12), "{}, {}".format(est.coef_, lr.coef_)
        assert np.all(np.abs(est.intercept_ - lr.intercept_) < 1e-12), "{}, {}".format(est.coef_, lr.intercept_)

        est = OLS(fit_intercept=False).fit(X, y, sample_weight=weights)
        lr = LinearRegression(fit_intercept=False).fit(X, y, sample_weight=weights)
        assert np.all(np.abs(est.coef_ - lr.coef_) < 1e-12), "{}, {}".format(est.coef_, lr.coef_)
        assert np.all(np.abs(est.intercept_ - lr.intercept_) < 1e-12), "{}, {}".format(est.coef_, lr.intercept_)

        n = 1000
        d = 3
        for p in np.arange(1, 4):
            X = np.random.binomial(1, .8, size=(n, d))
            T = np.random.binomial(1, .5 * X[:, 0] + .25, size=(n,))

            def true_effect(x):
                return np.hstack([x[:, [0]] + .5 + t for t in range(p)])
            y = np.zeros((n, p))
            y = true_effect(X) * T.reshape(-1, 1) + X[:, [0] * p] + \
                (0 * X[:, [0] * p] + 1) * np.random.normal(0, 1, size=(n, p))
            weights = np.random.uniform(0, 1, size=(X.shape[0]))

            est = OLS().fit(X, y)
            lr = LinearRegression().fit(X, y)
            assert np.all(np.abs(est.coef_ - lr.coef_) < 1e-12), "{}, {}".format(est.coef_, lr.coef_)
            assert np.all(np.abs(est.intercept_ - lr.intercept_) <
                          1e-12), "{}, {}".format(est.intercept_, lr.intercept_)

            est = OLS(fit_intercept=False).fit(X, y)
            lr = LinearRegression(fit_intercept=False).fit(X, y)
            assert np.all(np.abs(est.coef_ - lr.coef_) < 1e-12), "{}, {}".format(est.coef_, lr.coef_)
            assert np.all(np.abs(est.intercept_ - lr.intercept_) <
                          1e-12), "{}, {}".format(est.intercept_, lr.intercept_)

            est = OLS(fit_intercept=False).fit(X, y, sample_weight=weights)
            lr = LinearRegression(fit_intercept=False).fit(X, y, sample_weight=weights)
            assert np.all(np.abs(est.coef_ - lr.coef_) < 1e-12), "{}, {}".format(est.coef_, lr.coef_)
            assert np.all(np.abs(est.intercept_ - lr.intercept_) <
                          1e-12), "{}, {}".format(est.intercept_, lr.intercept_)

    def test_o_dtype(self):
        """ Testing that the models still work when the np arrays are of O dtype """
        np.random.seed(123)
        n = 1000
        d = 3

        X = np.random.normal(size=(n, d)).astype('O')
        y = np.random.normal(size=n).astype('O')

        est = OLS().fit(X, y)
        lr = LinearRegression().fit(X, y)
        assert np.all(np.abs(est.coef_ - lr.coef_) < 1e-12), "{}, {}".format(est.coef_, lr.coef_)
        assert np.all(np.abs(est.intercept_ - lr.intercept_) < 1e-12), "{}, {}".format(est.coef_, lr.intercept_)

        est = OLS(fit_intercept=False).fit(X, y)
        lr = LinearRegression(fit_intercept=False).fit(X, y)
        assert np.all(np.abs(est.coef_ - lr.coef_) < 1e-12), "{}, {}".format(est.coef_, lr.coef_)
        assert np.all(np.abs(est.intercept_ - lr.intercept_) < 1e-12), "{}, {}".format(est.coef_, lr.intercept_)

    def test_inference(self):
        """ Testing that we recover the expected standard errors and confidence intervals in a known example """

        # 1-d output
        d = 3
        X = np.vstack([np.eye(d)])
        y = X[:, 0]
        est = OLS(fit_intercept=False, cov_type="nonrobust").fit(X, y)
        assert np.all(np.abs(est.coef_ - [1, 0, 0]) <= 1e-12), "{}, {}".format(est.coef_, [1, 0, 0])
        assert np.all(np.abs(est.coef__interval() - np.array([[1, 0, 0], [1, 0, 0]])) <= 1e-12),\
            "{}, {}".format(est.coef__interval(), np.array([[1, 0, 0], [1, 0, 0]]))
        assert np.all(est.coef_stderr_ <= 1e-12)
        assert np.all(est._param_var <= 1e-12)

        d = 3
        X = np.vstack([np.eye(d), np.ones((1, d)), np.zeros((1, d))])
        y = X[:, 0]
        est = OLS(fit_intercept=True, cov_type="nonrobust").fit(X, y)
        assert np.all(np.abs(est.coef_ - np.array([1] + [0] * (d - 1))) <=
                      1e-12), "{}, {}".format(est.coef_, [1] + [0] * (d - 1))
        assert np.all(np.abs(est.coef__interval() - np.array([[1] + [0] * (d - 1), [1] + [0] * (d - 1)])) <= 1e-12),\
            "{}, {}".format(est.coef__interval(), np.array([[1] + [0] * (d - 1), [1] + [0] * (d - 1)]))
        assert np.all(est.coef_stderr_ <= 1e-12)
        assert np.all(est._param_var <= 1e-12)
        assert np.abs(est.intercept_) <= 1e-12
        assert np.all(np.abs(est.intercept__interval()) <= 1e-12)

        d = 3
        X = np.vstack([np.eye(d)])
        y = np.concatenate((X[:, 0] - 1, X[:, 0] + 1))
        X = np.vstack([X, X])
        est = OLS(fit_intercept=False, cov_type="nonrobust").fit(X, y)
        assert np.all(np.abs(est.coef_ - ([1] + [0] * (d - 1))) <=
                      1e-12), "{}, {}".format(est.coef_, [1] + [0] * (d - 1))
        assert np.all(np.abs(est.coef_stderr_ - np.array([1] * d)) <= 1e-12)
        assert np.all(np.abs(est.coef__interval()[0] -
                             np.array([scipy.stats.norm.ppf(.025, loc=1, scale=1)] +
                                      [scipy.stats.norm.ppf(.025, loc=0, scale=1)] * (d - 1))) <= 1e-12),\
            "{}, {}".format(est.coef__interval()[0], np.array([scipy.stats.norm.ppf(.025, loc=1, scale=1)] +
                                                              [scipy.stats.norm.ppf(.025, loc=0, scale=1)] * (d - 1)))
        assert np.all(np.abs(est.coef__interval()[1] -
                             np.array([scipy.stats.norm.ppf(.975, loc=1, scale=1)] +
                                      [scipy.stats.norm.ppf(.975, loc=0, scale=1)] * (d - 1))) <= 1e-12),\
            "{}, {}".format(est.coef__interval()[1], np.array([scipy.stats.norm.ppf(.975, loc=1, scale=1)] +
                                                              [scipy.stats.norm.ppf(.975, loc=0, scale=1)] * (d - 1)))

        # 2-d output
        d = 3
        p = 4
        X = np.vstack([np.eye(d)])
        y = np.vstack((X[:, [0] * p] - 1, X[:, [0] * p] + 1))
        X = np.vstack([X, X])
        est = OLS(fit_intercept=False, cov_type="nonrobust").fit(X, y)
        for t in range(p):
            assert np.all(np.abs(est.coef_[t] - ([1] + [0] * (d - 1))) <=
                          1e-12), "{}, {}".format(est.coef_[t], [1] + [0] * (d - 1))
            assert np.all(np.abs(est.coef_stderr_[t] - np.array([1] * d)) <= 1e-12), "{}".format(est.coef_stderr_[t])
            assert np.all(np.abs(est.coef__interval()[0][t] -
                                 np.array([scipy.stats.norm.ppf(.025, loc=1, scale=1)] +
                                          [scipy.stats.norm.ppf(.025, loc=0, scale=1)] * (d - 1))) <= 1e-12),\
                "{}, {}".format(est.coef__interval()[0][t],
                                np.array([scipy.stats.norm.ppf(.025, loc=1, scale=1)] +
                                         [scipy.stats.norm.ppf(.025, loc=0, scale=1)] * (d - 1)))
            assert np.all(np.abs(est.coef__interval()[1][t] -
                                 np.array([scipy.stats.norm.ppf(.975, loc=1, scale=1)] +
                                          [scipy.stats.norm.ppf(.975, loc=0, scale=1)] * (d - 1))) <= 1e-12),\
                "{}, {}".format(est.coef__interval()[1][t],
                                np.array([scipy.stats.norm.ppf(.975, loc=1, scale=1)] +
                                         [scipy.stats.norm.ppf(.975, loc=0, scale=1)] * (d - 1)))
            assert np.all(np.abs(est.intercept_[t]) <= 1e-12), "{}, {}".format(est.intercept_[t])
            assert np.all(np.abs(est.intercept_stderr_[t]) <= 1e-12), "{}".format(est.intercept_stderr_[t])
            assert np.all(np.abs(est.intercept__interval()[0][t]) <=
                          1e-12), "{}".format(est.intercept__interval()[0][t])

        d = 3
        p = 4
        X = np.vstack([np.eye(d), np.zeros((1, d))])
        y = np.vstack((X[:, [0] * p] - 1, X[:, [0] * p] + 1))
        X = np.vstack([X, X])
        est = OLS(fit_intercept=True, cov_type="nonrobust").fit(X, y)
        for t in range(p):
            assert np.all(np.abs(est.coef_[t] - ([1] + [0] * (d - 1))) <=
                          1e-12), "{}, {}".format(est.coef_[t], [1] + [0] * (d - 1))
            assert np.all(np.abs(est.coef_stderr_[t] - np.array([np.sqrt(2)] * d)) <=
                          1e-12), "{}".format(est.coef_stderr_[t])
            assert np.all(np.abs(est.coef__interval()[0][t] -
                                 np.array([scipy.stats.norm.ppf(.025, loc=1, scale=np.sqrt(2))] +
                                          [scipy.stats.norm.ppf(.025, loc=0, scale=np.sqrt(2))] * (d - 1))) <= 1e-12),\
                "{}, {}".format(est.coef__interval()[0][t],
                                np.array([scipy.stats.norm.ppf(.025, loc=1, scale=np.sqrt(2))] +
                                         [scipy.stats.norm.ppf(.025, loc=0, scale=np.sqrt(2))] * (d - 1)))
            assert np.all(np.abs(est.coef__interval()[1][t] -
                                 np.array([scipy.stats.norm.ppf(.975, loc=1, scale=np.sqrt(2))] +
                                          [scipy.stats.norm.ppf(.975, loc=0, scale=np.sqrt(2))] * (d - 1))) <= 1e-12),\
                "{}, {}".format(est.coef__interval()[1][t],
                                np.array([scipy.stats.norm.ppf(.975, loc=1, scale=np.sqrt(2))] +
                                         [scipy.stats.norm.ppf(.975, loc=0, scale=np.sqrt(2))] * (d - 1)))
            assert np.all(np.abs(est.intercept_[t]) <= 1e-12), "{}, {}".format(est.intercept_[t])
            assert np.all(np.abs(est.intercept_stderr_[t] - 1) <= 1e-12), "{}".format(est.intercept_stderr_[t])
            assert np.all(np.abs(est.intercept__interval()[0][t] -
                                 scipy.stats.norm.ppf(.025, loc=0, scale=1)) <= 1e-12),\
                "{}, {}".format(est.intercept__interval()[0][t], scipy.stats.norm.ppf(.025, loc=0, scale=1))

    def test_comp_with_statsmodels(self):
        """ Comparing with confidence intervals and standard errors of statsmodels in the un-weighted case """
        np.random.seed(123)

        # Single dimensional output y
        n = 1000
        d = 3
        X = np.random.binomial(1, .8, size=(n, d))
        T = np.random.binomial(1, .5 * X[:, 0] + .25, size=(n,))
        weight = 0.5 * X[:, 1] + 1.2 * X[:, 2]

        def true_effect(x):
            return x[:, 0] + .5
        y = true_effect(X) * T + X[:, 0] + X[:, 2] + np.random.normal(0, 1, size=(n,))
        X_test = np.unique(np.random.binomial(1, .5, size=(n, d)), axis=0)
        for fit_intercept in [True, False]:
            for cov_type in ['nonrobust', 'HC0', 'HC1']:
                est = OLS(fit_intercept=fit_intercept, cov_type=cov_type).fit(X, y)
                lr = StatsModelsOLS(fit_intercept=fit_intercept, fit_args={
                                    'cov_type': cov_type, 'use_t': False}).fit(X, y)
                _compare_classes(est, lr, X_test)
                # compare with weight
                est = OLS(fit_intercept=fit_intercept, cov_type=cov_type).fit(X, y, sample_weight=weight)
                lr = StatsModelsOLS(fit_intercept=fit_intercept, fit_args={
                                    'cov_type': cov_type, 'use_t': False}).fit(X, y, sample_weight=weight)
                _compare_classes(est, lr, X_test)

        n = 1000
        d = 3
        X = np.random.normal(0, 1, size=(n, d))
        y = X[:, 0] + X[:, 2] + np.random.normal(0, 1, size=(n,))
        X_test = np.unique(np.random.binomial(1, .5, size=(n, d)), axis=0)
        weight = abs(0.5 * X[:, 1] + 1.2 * X[:, 2])
        for fit_intercept in [True, False]:
            for cov_type in ['nonrobust', 'HC0', 'HC1']:
                est = OLS(fit_intercept=fit_intercept, cov_type=cov_type).fit(X, y)
                lr = StatsModelsOLS(fit_intercept=fit_intercept, fit_args={
                                    'cov_type': cov_type, 'use_t': False}).fit(X, y)
                _compare_classes(est, lr, X_test)
                # compare with weight
                est = OLS(fit_intercept=fit_intercept, cov_type=cov_type).fit(X, y, sample_weight=weight)
                lr = StatsModelsOLS(fit_intercept=fit_intercept, fit_args={
                                    'cov_type': cov_type, 'use_t': False}).fit(X, y, sample_weight=weight)
                _compare_classes(est, lr, X_test)

        d = 3
        X = np.vstack([np.eye(d)])
        y = np.concatenate((X[:, 0] - 1, X[:, 0] + 1))
        X = np.vstack([X, X])
        X_test = np.unique(np.random.binomial(1, .5, size=(n, d)), axis=0)
        weight = np.random.uniform(0, 1, size=(X.shape[0],))
        for cov_type in ['nonrobust', 'HC0', 'HC1']:
            for alpha in [.01, .05, .1]:
                _compare_classes(OLS(fit_intercept=False, cov_type=cov_type).fit(X, y),
                                 StatsModelsOLS(fit_intercept=False, fit_args={
                                                'cov_type': cov_type, 'use_t': False}).fit(X, y),
                                 X_test, alpha=alpha)
                # compare with weight
                _compare_classes(OLS(fit_intercept=False, cov_type=cov_type).fit(X, y, sample_weight=weight),
                                 StatsModelsOLS(fit_intercept=False, fit_args={
                                                'cov_type': cov_type, 'use_t': False}).fit(X, y, sample_weight=weight),
                                 X_test, alpha=alpha)

        d = 3
        X = np.vstack([np.eye(d), np.ones((1, d)), np.zeros((1, d))])
        y = np.concatenate((X[:, 0] - 1, X[:, 0] + 1))
        X = np.vstack([X, X])
        X_test = np.unique(np.random.binomial(1, .5, size=(n, d)), axis=0)
        weight = np.random.uniform(0, 1, size=(X.shape[0],))
        for cov_type in ['nonrobust', 'HC0', 'HC1']:
            _compare_classes(OLS(fit_intercept=True, cov_type=cov_type).fit(X, y),
                             StatsModelsOLS(fit_intercept=True,
                                            fit_args={'cov_type': cov_type, 'use_t': False}).fit(X, y), X_test)
            # compare with weight
            _compare_classes(OLS(fit_intercept=True, cov_type=cov_type).fit(X, y, sample_weight=weight),
                             StatsModelsOLS(fit_intercept=True, fit_args={
                                 'cov_type': cov_type, 'use_t': False}).fit(X, y, sample_weight=weight),
                             X_test)

        # Multi-dimensional output y
        n = 1000
        d = 3
        for p in np.arange(1, 4):
            X = np.random.binomial(1, .8, size=(n, d))
            T = np.random.binomial(1, .5 * X[:, 0] + .25, size=(n,))

            def true_effect(x):
                return np.hstack([x[:, [0]] + .5 + t for t in range(p)])
            y = np.zeros((n, p))
            y = true_effect(X) * T.reshape(-1, 1) + X[:, [0] * p] + \
                (0 * X[:, [0] * p] + 1) * np.random.normal(0, 1, size=(n, p))
            weight = 0.5 * X[:, 1] + 1.2 * X[:, 2]

            for cov_type in ['nonrobust', 'HC0', 'HC1']:
                for fit_intercept in [True, False]:
                    for alpha in [.01, .05, .2]:
                        est = OLS(fit_intercept=fit_intercept, cov_type=cov_type).fit(X, y, sample_weight=weight)
                        lr = [StatsModelsOLS(fit_intercept=fit_intercept, fit_args={
                                             'cov_type': cov_type, 'use_t': False}).fit(X, y[:, t],
                                                                                        sample_weight=weight)
                              for t in range(p)]
                        for t in range(p):
                            assert np.all(np.abs(est.coef_[t] - lr[t].coef_) < 1e-12),\
                                "{}, {}, {}: {}, {}".format(cov_type, fit_intercept, t, est.coef_[t], lr[t].coef_)
                            assert np.all(np.abs(np.array(est.coef__interval(alpha=alpha))[:, t] -
                                                 lr[t].coef__interval(alpha=alpha)) < 1e-12),\
                                "{}, {}, {}: {} vs {}".format(cov_type, fit_intercept, t,
                                                              np.array(est.coef__interval(alpha=alpha))[:, t],
                                                              lr[t].coef__interval(alpha=alpha))
                            assert np.all(np.abs(est.intercept_[t] - lr[t].intercept_) < 1e-12),\
                                "{}, {}, {}: {} vs {}".format(cov_type, fit_intercept, t,
                                                              est.intercept_[t], lr[t].intercept_)
                            assert np.all(np.abs(np.array(est.intercept__interval(alpha=alpha))[:, t] -
                                                 lr[t].intercept__interval(alpha=alpha)) < 1e-12),\
                                "{}, {}, {}: {} vs {}".format(cov_type, fit_intercept, t,
                                                              np.array(est.intercept__interval(alpha=alpha))[:, t],
                                                              lr[t].intercept__interval(alpha=alpha))
                            assert np.all(np.abs(est.predict(X_test)[:, t] - lr[t].predict(X_test)) < 1e-12),\
                                "{}, {}, {}: {} vs {}".format(cov_type, fit_intercept, t, est.predict(X_test)[
                                                              :, t], lr[t].predict(X_test))
                            assert np.all(np.abs(np.array(est.predict_interval(X_test, alpha=alpha))[:, :, t] -
                                                 lr[t].predict_interval(X_test, alpha=alpha)) < 1e-12),\
                                "{}, {}, {}: {} vs {}".format(cov_type, fit_intercept, t,
                                                              np.array(est.predict_interval(X_test,
                                                                                            alpha=alpha))[:, :, t],
                                                              lr[t].predict_interval(X_test, alpha=alpha))

    def test_sum_vs_original(self):
        """ Testing that the summarized version gives the same results as the non-summarized."""
        np.random.seed(123)
        # 1-d y
        n = 100
        p = 1
        d = 5
        X_test = np.random.binomial(1, .5, size=(100, d))

        X = np.random.binomial(1, .8, size=(n, d))
        y = X[:, [0] * p] + (1 * X[:, [0]] + 1) * np.random.normal(0, 1, size=(n, p))
        y = y.flatten()
        w = X[:, 0] * 0.5 + X[:, 1] * 1.2

        (X1, X2, y1, y2, w1, w2,
         X_final_first, X_final_sec, y_sum_first, y_sum_sec, w_sum_first, w_sum_sec, n_sum_first, n_sum_sec,
         var_first, var_sec) = _summarize(X, y, w)
        X = np.vstack([X1, X2])
        y = np.concatenate((y1, y2))
        w = np.concatenate((w1, w2))
        X_final = np.vstack([X_final_first, X_final_sec])
        y_sum = np.concatenate((y_sum_first, y_sum_sec))
        w_sum = np.concatenate((w_sum_first, w_sum_sec))
        n_sum = np.concatenate((n_sum_first, n_sum_sec))
        var_sum = np.concatenate((var_first, var_sec))

        for cov_type in ['nonrobust', 'HC0', 'HC1']:
            for fit_intercept in [True, False]:
                for alpha in [.01, .05, .2]:
                    _compare_classes(OLS(fit_intercept=fit_intercept,
                                         cov_type=cov_type).fit(X, y),
                                     OLS(fit_intercept=fit_intercept,
                                         cov_type=cov_type).fit(X_final, y_sum,
                                                                freq_weight=n_sum, sample_var=var_sum),
                                     X_test, alpha=alpha)
                    _compare_classes(StatsModelsOLS(fit_intercept=fit_intercept,
                                                    fit_args={'cov_type': cov_type, 'use_t': False}).fit(X, y),
                                     OLS(fit_intercept=fit_intercept,
                                         cov_type=cov_type).fit(X_final, y_sum,
                                                                freq_weight=n_sum, sample_var=var_sum),
                                     X_test, alpha=alpha)
                    # compare when both sample_var and sample_weight exist
                    _compare_classes(OLS(fit_intercept=fit_intercept,
                                         cov_type=cov_type).fit(X, y, sample_weight=w),
                                     OLS(fit_intercept=fit_intercept,
                                         cov_type=cov_type).fit(X_final, y_sum, sample_weight=w_sum,
                                                                freq_weight=n_sum, sample_var=var_sum),
                                     X_test, alpha=alpha)
                    _compare_classes(StatsModelsOLS(fit_intercept=fit_intercept,
                                                    fit_args={'cov_type': cov_type,
                                                              'use_t': False}).fit(X, y, sample_weight=w),
                                     OLS(fit_intercept=fit_intercept,
                                         cov_type=cov_type).fit(X_final, y_sum, sample_weight=w_sum,
                                                                freq_weight=n_sum, sample_var=var_sum),
                                     X_test, alpha=alpha)

        # multi-d y
        n = 100
        for d in [1, 5]:
            for p in [1, 5]:
                X_test = np.random.binomial(1, .5, size=(100, d))

                X = np.random.binomial(1, .8, size=(n, d))
                y = X[:, [0] * p] + (1 * X[:, [0]] + 1) * np.random.normal(0, 1, size=(n, p))
                w = X[:, 0] * 1.2

                (X1, X2, y1, y2, w1, w2,
                 X_final_first, X_final_sec, y_sum_first, y_sum_sec, w_sum_first, w_sum_sec, n_sum_first, n_sum_sec,
                 var_first, var_sec) = _summarize(X, y, w)
                X = np.vstack([X1, X2])
                y = np.concatenate((y1, y2))
                w = np.concatenate((w1, w2))
                X_final = np.vstack([X_final_first, X_final_sec])
                y_sum = np.concatenate((y_sum_first, y_sum_sec))
                w_sum = np.concatenate((w_sum_first, w_sum_sec))
                n_sum = np.concatenate((n_sum_first, n_sum_sec))
                var_sum = np.concatenate((var_first, var_sec))

                for cov_type in ['nonrobust', 'HC0', 'HC1']:
                    for fit_intercept in [True, False]:
                        for alpha in [.01, .05, .2]:
                            _compare_classes(OLS(fit_intercept=fit_intercept, cov_type=cov_type).fit(X, y),
                                             OLS(fit_intercept=fit_intercept,
                                                 cov_type=cov_type).fit(X_final, y_sum,
                                                                        freq_weight=n_sum,
                                                                        sample_var=var_sum),
                                             X_test, alpha=alpha)
                            # compare when both sample_var and sample_weight exist
                            _compare_classes(OLS(fit_intercept=fit_intercept,
                                                 cov_type=cov_type).fit(X, y, sample_weight=w),
                                             OLS(fit_intercept=fit_intercept,
                                                 cov_type=cov_type).fit(X_final, y_sum, sample_weight=w_sum,
                                                                        freq_weight=n_sum,
                                                                        sample_var=var_sum),
                                             X_test, alpha=alpha)

    def test_dml_sum_vs_original(self):
        """ Testing that the summarized version of DML gives the same results as the non-summarized. """
        np.random.seed(123)

        n = 1000
        for d in [1, 5]:
            for p in [1, 5]:
                for cov_type in ['nonrobust', 'HC0', 'HC1']:
                    for alpha in [.01, .05, .2]:
                        X = np.random.binomial(1, .8, size=(n, d))
                        T = np.random.binomial(1, .5 * X[:, 0] + .25, size=(n,))
                        w = X[:, 0] * 1.2

                        def true_effect(x):
                            return np.hstack([x[:, [0]] + t for t in range(p)])
                        y = true_effect(X) * T.reshape(-1, 1) + X[:, [0] * p] + \
                            (1 * X[:, [0]] + 1) * np.random.normal(0, 1, size=(n, p))
                        if p == 1:
                            y = y.flatten()
                        X_test = np.random.binomial(1, .5, size=(100, d))

                        XT = np.hstack([X, T.reshape(-1, 1)])
                        (X1, X2, y1, y2, w1, w2,
                            X_final_first, X_final_sec, y_sum_first, y_sum_sec, w_sum_first, w_sum_sec, n_sum_first,
                            n_sum_sec, var_first, var_sec) = _summarize(XT, y, w)
                        X = np.vstack([X1, X2])
                        y = np.concatenate((y1, y2))
                        w = np.concatenate((w1, w2))
                        X_final = np.vstack([X_final_first, X_final_sec])
                        y_sum = np.concatenate((y_sum_first, y_sum_sec))
                        w_sum = np.concatenate((w_sum_first, w_sum_sec))
                        n_sum = np.concatenate((n_sum_first, n_sum_sec))
                        var_sum = np.concatenate((var_first, var_sec))
                        first_half_sum = len(y_sum_first)
                        first_half = len(y1)

                        class SplitterSum:
                            def __init__(self):
                                return

                            def split(self, X, T):
                                return [(np.arange(0, first_half_sum), np.arange(first_half_sum, X.shape[0])),
                                        (np.arange(first_half_sum, X.shape[0]), np.arange(0, first_half_sum))]

                        class Splitter:
                            def __init__(self):
                                return

                            def split(self, X, T):
                                return [(np.arange(0, first_half), np.arange(first_half, X.shape[0])),
                                        (np.arange(first_half, X.shape[0]), np.arange(0, first_half))]
                        for first_stage_model in [LinearRegression(),
                                                  WeightedLasso(alpha=0.01, fit_intercept=True,
                                                                tol=1e-12, random_state=123),
                                                  RandomForestRegressor(n_estimators=10, bootstrap=False,
                                                                        random_state=123)]:
                            est = LinearDML(
                                model_y=first_stage_model,
                                model_t=first_stage_model,
                                cv=SplitterSum(),
                                linear_first_stages=False)
                            lr = LinearDML(
                                model_y=first_stage_model,
                                model_t=first_stage_model,
                                cv=Splitter(),
                                linear_first_stages=False)

                            est.fit(y_sum,
                                    X_final[:, -1], X=X_final[:, :-1],
                                    W=None, freq_weight=n_sum,
                                    sample_var=var_sum,
                                    inference=StatsModelsInference(cov_type=cov_type))
                            lr.fit(y, X[:, -1], X=X[:, :-1], W=None,
                                   inference=StatsModelsInference(cov_type=cov_type))
                            _compare_dml_classes(est, lr, X_test, alpha=alpha, tol=1e-8)

                            # compare when both sample_var and sample_weight exist
                            est.fit(y_sum,
                                    X_final[:, -1], X=X_final[:, :-1],
                                    W=None, sample_weight=w_sum, freq_weight=n_sum,
                                    sample_var=var_sum,
                                    inference=StatsModelsInference(cov_type=cov_type))
                            lr.fit(y, X[:, -1], X=X[:, :-1], W=None, sample_weight=w,
                                   inference=StatsModelsInference(cov_type=cov_type))
                            _compare_dml_classes(est, lr, X_test, alpha=alpha, tol=1e-8)

    def test_nonparamdml_sum_vs_original(self):
        """ Testing that the summarized version of DML gives the same results as the non-summarized. """
        np.random.seed(123)

        n = 1000
        for d in [1, 5]:
            for p in [1, 5]:
                for alpha in [.01, .05, .2]:
                    X = np.random.binomial(1, .8, size=(n, d))
                    T = np.random.binomial(1, .5 * X[:, 0] + .25, size=(n,))
                    w = X[:, 0] * 1.2

                    def true_effect(x):
                        return np.hstack([x[:, [0]] + t for t in range(p)])
                    y = true_effect(X) * T.reshape(-1, 1) + X[:, [0] * p] + \
                        (1 * X[:, [0]] + 1) * np.random.normal(0, 1, size=(n, p))
                    if p == 1:
                        y = y.flatten()
                    X_test = np.random.binomial(1, .5, size=(100, d))

                    XT = np.hstack([X, T.reshape(-1, 1)])
                    (X1, X2, y1, y2, w1, w2,
                        X_final_first, X_final_sec, y_sum_first, y_sum_sec, w_sum_first, w_sum_sec, n_sum_first,
                        n_sum_sec, var_first, var_sec) = _summarize(XT, y, w)
                    X = np.vstack([X1, X2])
                    y = np.concatenate((y1, y2))
                    w = np.concatenate((w1, w2))
                    X_final = np.vstack([X_final_first, X_final_sec])
                    y_sum = np.concatenate((y_sum_first, y_sum_sec))
                    w_sum = np.concatenate((w_sum_first, w_sum_sec))
                    n_sum = np.concatenate((n_sum_first, n_sum_sec))
                    var_sum = np.concatenate((var_first, var_sec))
                    first_half_sum = len(y_sum_first)
                    first_half = len(y1)

                    class SplitterSum:
                        def __init__(self):
                            return

                        def split(self, X, T):
                            return [(np.arange(0, first_half_sum), np.arange(first_half_sum, X.shape[0])),
                                    (np.arange(first_half_sum, X.shape[0]), np.arange(0, first_half_sum))]

                    class Splitter:
                        def __init__(self):
                            return

                        def split(self, X, T):
                            return [(np.arange(0, first_half), np.arange(first_half, X.shape[0])),
                                    (np.arange(first_half, X.shape[0]), np.arange(0, first_half))]
                    for first_stage_model in [LinearRegression(),
                                              WeightedLasso(alpha=0.01, fit_intercept=True,
                                                            tol=1e-12, random_state=123),
                                              RandomForestRegressor(n_estimators=10, bootstrap=False,
                                                                    random_state=123)]:
                        est = NonParamDML(
                            model_y=first_stage_model,
                            model_t=first_stage_model,
                            model_final=OLS(),
                            cv=SplitterSum()).fit(y_sum,
                                                  X_final[:, -1], X=X_final[:, :-1],
                                                  W=None, freq_weight=n_sum,
                                                  sample_var=var_sum,
                                                  inference="auto")

                        lr = NonParamDML(
                            model_y=first_stage_model,
                            model_t=first_stage_model,
                            model_final=OLS(),
                            cv=Splitter()).fit(y, X[:, -1], X=X[:, :-1], W=None,
                                               inference="auto")
                        _compare_classes(est.model_final_, lr.model_final_, X_test, alpha=alpha, tol=1e-10)

                        # compare when both sample_var and sample_weight exist
                        est.fit(y_sum,
                                X_final[:, -1], X=X_final[:, :-1],
                                W=None, sample_weight=w_sum, freq_weight=n_sum,
                                sample_var=var_sum,
                                inference="auto")
                        lr.fit(y, X[:, -1], X=X[:, :-1], W=None, sample_weight=w,
                               inference="auto")
                        _compare_classes(est.model_final_, lr.model_final_, X_test, alpha=alpha, tol=1e-10)

    def test_dr_sum_vs_original(self):
        """ Testing that the summarized version of DR gives the same results as the non-summarized. """
        np.random.seed(123)

        n = 1000
        for d in [1, 5]:
            for cov_type in ['nonrobust', 'HC0', 'HC1']:
                for alpha in [.01, .05, .2]:
                    X = np.random.binomial(1, .8, size=(n, d))
                    T = np.random.binomial(1, .5 * X[:, 0] + .25, size=(n,))
                    w = X[:, 0] * 1.2

                    def true_effect(x):
                        return x[:, [0]]
                    y = true_effect(X) * T.reshape(-1, 1) + X[:, [0] * 1] + \
                        (1 * X[:, [0]] + 1) * np.random.normal(0, 1, size=(n, 1))
                    y = y.flatten()
                    X_test = np.random.binomial(1, .5, size=(100, d))

                    XT = np.hstack([X, T.reshape(-1, 1)])
                    (X1, X2, y1, y2, w1, w2,
                        X_final_first, X_final_sec, y_sum_first, y_sum_sec, w_sum_first, w_sum_sec, n_sum_first,
                        n_sum_sec, var_first, var_sec) = _summarize(XT, y, w)
                    X = np.vstack([X1, X2])
                    y = np.concatenate((y1, y2))
                    w = np.concatenate((w1, w2))
                    X_final = np.vstack([X_final_first, X_final_sec])
                    y_sum = np.concatenate((y_sum_first, y_sum_sec))
                    w_sum = np.concatenate((w_sum_first, w_sum_sec))
                    n_sum = np.concatenate((n_sum_first, n_sum_sec))
                    var_sum = np.concatenate((var_first, var_sec))
                    first_half_sum = len(y_sum_first)
                    first_half = len(y1)

                    class SplitterSum:
                        def __init__(self):
                            return

                        def split(self, X, T):
                            return [(np.arange(0, first_half_sum), np.arange(first_half_sum, X.shape[0])),
                                    (np.arange(first_half_sum, X.shape[0]), np.arange(0, first_half_sum))]

                    class Splitter:
                        def __init__(self):
                            return

                        def split(self, X, T):
                            return [(np.arange(0, first_half), np.arange(first_half, X.shape[0])),
                                    (np.arange(first_half, X.shape[0]), np.arange(0, first_half))]
                    for model_regression in [LinearRegression(),
                                             WeightedLasso(alpha=0.01, fit_intercept=True,
                                                           tol=1e-12, random_state=123),
                                             RandomForestRegressor(n_estimators=10, bootstrap=False,
                                                                   max_depth=3,
                                                                   random_state=123)]:
                        model_propensity = LogisticRegression(random_state=123)
                        est = LinearDRLearner(
                            model_regression=model_regression,
                            model_propensity=model_propensity,
                            cv=SplitterSum())
                        lr = LinearDRLearner(
                            model_regression=model_regression,
                            model_propensity=model_propensity,
                            cv=Splitter())

                        est.fit(y_sum,
                                X_final[:, -1], X=X_final[:, :-1],
                                W=None, freq_weight=n_sum,
                                sample_var=var_sum,
                                inference=StatsModelsInferenceDiscrete(cov_type=cov_type))
                        lr.fit(y, X[:, -1], X=X[:, :-1], W=None,
                               inference=StatsModelsInferenceDiscrete(cov_type=cov_type))
                        _compare_dr_classes(est, lr, X_test, alpha=alpha, tol=1e-8)

                        # compare when both sample_var and sample_weight exist
                        est.fit(y_sum,
                                X_final[:, -1], X=X_final[:, :-1],
                                W=None, sample_weight=w_sum, freq_weight=n_sum,
                                sample_var=var_sum,
                                inference=StatsModelsInferenceDiscrete(cov_type=cov_type))
                        lr.fit(y, X[:, -1], X=X[:, :-1], W=None, sample_weight=w,
                               inference=StatsModelsInferenceDiscrete(cov_type=cov_type))
                        _compare_dr_classes(est, lr, X_test, alpha=alpha, tol=1e-8)

    def test_lineardriv_sum_vs_original(self):
        """ Testing that the summarized version of DR gives the same results as the non-summarized. """
        np.random.seed(123)

        n = 1000
        for d in [1, 5]:
            for cov_type in ['nonrobust', 'HC0', 'HC1']:
                for alpha in [.01, .05, .2]:
                    X = np.random.binomial(1, .8, size=(n, d))
                    Z = np.random.binomial(1, .5 * X[:, 0] + .25, size=(n,))
                    T = np.random.binomial(1, .8 * Z + .1, size=(n,))
                    w = 0.5 + X[:, 0] * 1.2

                    def true_effect(x):
                        return x[:, [0]]
                    y = true_effect(X) * T.reshape(-1, 1) + X[:, [0] * 1] + \
                        (1 * X[:, [0]] + 1) * np.random.normal(0, 1, size=(n, 1))
                    y = y.flatten()
                    X_test = np.random.binomial(1, .5, size=(100, d))

                    XTZ = np.hstack([X, T.reshape(-1, 1), Z.reshape(-1, 1)])
                    (X1, X2, y1, y2, w1, w2,
                        X_final_first, X_final_sec, y_sum_first, y_sum_sec, w_sum_first, w_sum_sec, n_sum_first,
                        n_sum_sec, var_first, var_sec) = _summarize(XTZ, y, w)
                    X = np.vstack([X1, X2])
                    y = np.concatenate((y1, y2))
                    w = np.concatenate((w1, w2))
                    X_final = np.vstack([X_final_first, X_final_sec])
                    y_sum = np.concatenate((y_sum_first, y_sum_sec))
                    w_sum = np.concatenate((w_sum_first, w_sum_sec))
                    n_sum = np.concatenate((n_sum_first, n_sum_sec))
                    var_sum = np.concatenate((var_first, var_sec))
                    first_half_sum = len(y_sum_first)
                    first_half = len(y1)

                    class SplitterSum:
                        def __init__(self):
                            return

                        def split(self, X, T):
                            return [(np.arange(0, first_half_sum), np.arange(first_half_sum, X.shape[0])),
                                    (np.arange(first_half_sum, X.shape[0]), np.arange(0, first_half_sum))]

                    class Splitter:
                        def __init__(self):
                            return

                        def split(self, X, T):
                            return [(np.arange(0, first_half), np.arange(first_half, X.shape[0])),
                                    (np.arange(first_half, X.shape[0]), np.arange(0, first_half))]
                    for model_regression in [LinearRegression(),
                                             WeightedLasso(alpha=0.01, fit_intercept=True,
                                                           tol=1e-12, random_state=123),
                                             RandomForestRegressor(n_estimators=10, bootstrap=False,
                                                                   max_depth=3,
                                                                   random_state=123)]:
                        model_propensity = LogisticRegression(random_state=123)
                        est = LinearDRIV(
                            model_y_xw=model_regression,
                            model_t_xw=model_propensity,
                            model_z_xw=model_propensity,
                            model_tz_xw=model_propensity,
                            flexible_model_effect=StatsModelsLinearRegression(fit_intercept=False),
                            discrete_instrument=True,
                            discrete_treatment=True,
                            cv=SplitterSum())

                        lr = LinearDRIV(
                            model_y_xw=model_regression,
                            model_t_xw=model_propensity,
                            model_z_xw=model_propensity,
                            model_tz_xw=model_propensity,
                            flexible_model_effect=StatsModelsLinearRegression(fit_intercept=False),
                            discrete_instrument=True,
                            discrete_treatment=True,
                            cv=Splitter())

                        est.fit(y_sum,
                                X_final[:, -2], Z=X_final[:, -1], X=X_final[:, :-2],
                                W=None, freq_weight=n_sum,
                                sample_var=var_sum,
                                inference=StatsModelsInference(cov_type=cov_type))
                        lr.fit(y, X[:, -2], Z=X[:, -1], X=X[:, :-2], W=None,
                               inference=StatsModelsInference(cov_type=cov_type))
                        _compare_dml_classes(est, lr, X_test, alpha=alpha, tol=1e-8)

                        # compare when both sample_var and sample_weight exist
                        est.fit(y_sum,
                                X_final[:, -2], Z=X_final[:, -1], X=X_final[:, :-2],
                                W=None, sample_weight=w_sum, freq_weight=n_sum,
                                sample_var=var_sum,
                                inference=StatsModelsInference(cov_type=cov_type))
                        lr.fit(y, X[:, -2], Z=X[:, -1], X=X[:, :-2], W=None, sample_weight=w,
                               inference=StatsModelsInference(cov_type=cov_type))
                        _compare_dml_classes(est, lr, X_test, alpha=alpha, tol=1e-8)

    def test_dmliv_sum_vs_original(self):
        """ Testing that the summarized version of DR gives the same results as the non-summarized. """
        np.random.seed(123)

        n = 1000
        for d in [1, 5]:
            for cov_type in ['nonrobust', 'HC0', 'HC1']:
                for alpha in [.01, .05, .2]:
                    X = np.random.binomial(1, .8, size=(n, d))
                    Z = np.random.binomial(1, .5 * X[:, 0] + .25, size=(n,))
                    T = np.random.binomial(1, .8 * Z + .1, size=(n,))
                    w = 0.5 + X[:, 0] * 1.2

                    def true_effect(x):
                        return x[:, [0]]
                    y = true_effect(X) * T.reshape(-1, 1) + X[:, [0] * 1] + \
                        (1 * X[:, [0]] + 1) * np.random.normal(0, 1, size=(n, 1))
                    y = y.flatten()
                    X_test = np.random.binomial(1, .5, size=(100, d))

                    XTZ = np.hstack([X, T.reshape(-1, 1), Z.reshape(-1, 1)])
                    (X1, X2, y1, y2, w1, w2,
                        X_final_first, X_final_sec, y_sum_first, y_sum_sec, w_sum_first, w_sum_sec, n_sum_first,
                        n_sum_sec, var_first, var_sec) = _summarize(XTZ, y, w)
                    X = np.vstack([X1, X2])
                    y = np.concatenate((y1, y2))
                    w = np.concatenate((w1, w2))
                    X_final = np.vstack([X_final_first, X_final_sec])
                    y_sum = np.concatenate((y_sum_first, y_sum_sec))
                    w_sum = np.concatenate((w_sum_first, w_sum_sec))
                    n_sum = np.concatenate((n_sum_first, n_sum_sec))
                    var_sum = np.concatenate((var_first, var_sec))
                    first_half_sum = len(y_sum_first)
                    first_half = len(y1)

                    class SplitterSum:
                        def __init__(self):
                            return

                        def split(self, X, T):
                            return [(np.arange(0, first_half_sum), np.arange(first_half_sum, X.shape[0])),
                                    (np.arange(first_half_sum, X.shape[0]), np.arange(0, first_half_sum))]

                    class Splitter:
                        def __init__(self):
                            return

                        def split(self, X, T):
                            return [(np.arange(0, first_half), np.arange(first_half, X.shape[0])),
                                    (np.arange(first_half, X.shape[0]), np.arange(0, first_half))]
                    for model_regression in [LinearRegression(),
                                             WeightedLasso(alpha=0.01, fit_intercept=True,
                                                           tol=1e-12, random_state=123),
                                             RandomForestRegressor(n_estimators=10, bootstrap=False,
                                                                   max_depth=3,
                                                                   random_state=123)]:
                        model_propensity = LogisticRegression(random_state=123)
                        est = DMLIV(
                            model_y_xw=model_regression,
                            model_t_xw=model_propensity,
                            model_t_xwz=model_propensity,
                            discrete_instrument=True,
                            discrete_treatment=True,
                            fit_cate_intercept=False,
                            cv=SplitterSum())

                        lr = DMLIV(
                            model_y_xw=model_regression,
                            model_t_xw=model_propensity,
                            model_t_xwz=model_propensity,
                            discrete_instrument=True,
                            discrete_treatment=True,
                            fit_cate_intercept=False,
                            cv=Splitter())

                        est.fit(y_sum,
                                X_final[:, -2], Z=X_final[:, -1], X=X_final[:, :-2],
                                W=None, freq_weight=n_sum,
                                sample_var=var_sum,
                                inference=StatsModelsInference(cov_type=cov_type))
                        lr.fit(y, X[:, -2], Z=X[:, -1], X=X[:, :-2], W=None,
                               inference=StatsModelsInference(cov_type=cov_type))
                        _compare_classes(est.model_final_, lr.model_final_, X_test, alpha=alpha, tol=1e-8)

                        # compare when both sample_var and sample_weight exist
                        est.fit(y_sum,
                                X_final[:, -2], Z=X_final[:, -1], X=X_final[:, :-2],
                                W=None, sample_weight=w_sum, freq_weight=n_sum,
                                sample_var=var_sum,
                                inference=StatsModelsInference(cov_type=cov_type))
                        lr.fit(y, X[:, -2], Z=X[:, -1], X=X[:, :-2], W=None, sample_weight=w,
                               inference=StatsModelsInference(cov_type=cov_type))
                        _compare_classes(est.model_final_, lr.model_final_, X_test, alpha=alpha, tol=1e-8)

    def test_dml_multi_dim_treatment_outcome(self):
        """ Testing that the summarized and unsummarized version of DML gives the correct (known results). """
        np.random.seed(123)
        n = 100000
        precision = .01
        precision_int = .0001
        with np.printoptions(formatter={'float': '{:.4f}'.format}, suppress=True):
            for d in [2, 5]:  # n_feats + n_controls
                for d_x in [1]:  # n_feats
                    for p in [1, 5]:  # n_outcomes
                        for q in [1, 5]:  # n_treatments
                            X = np.random.binomial(1, .5, size=(n, d))
                            T = np.hstack([np.random.binomial(1, .5 + .2 * (2 * X[:, [1]] - 1)) for _ in range(q)])

                            def true_effect(x, i):
                                return np.hstack([x[:, [0]] + 10 * t + i for t in range(p)])
                            y = np.sum((true_effect(X, i) * T[:, [i]] for i in range(q)), axis=0) + X[:, [0] * p]
                            if p == 1:
                                y = y.flatten()
                            est = LinearDML(model_y=LinearRegression(),
                                            model_t=LinearRegression(),
                                            linear_first_stages=False)
                            est.fit(y, T, X=X[:, :d_x], W=X[:, d_x:],
                                    inference=StatsModelsInference(cov_type='nonrobust'))
                            intercept = est.intercept_.reshape((p, q))
                            lower_int, upper_int = est.intercept__interval(alpha=.001)
                            lower_int = lower_int.reshape((p, q))
                            upper_int = upper_int.reshape((p, q))
                            coef = est.coef_.reshape(p, q, d_x)
                            lower, upper = est.coef__interval(alpha=.001)
                            lower = lower.reshape(p, q, d_x)
                            upper = upper.reshape(p, q, d_x)
                            for i in range(p):
                                for j in range(q):
                                    np.testing.assert_allclose(intercept[i, j], 10 * i + j, rtol=0, atol=precision)
                                    np.testing.assert_array_less(lower_int[i, j], 10 * i + j + precision_int)
                                    np.testing.assert_array_less(10 * i + j - precision_int, upper_int[i, j])
                                    np.testing.assert_allclose(coef[i, j, 0], 1, atol=precision)
                                    np.testing.assert_array_less(lower[i, j, 0], 1)
                                    np.testing.assert_array_less(1, upper[i, j, 0])
                                    np.testing.assert_allclose(coef[i, j, 1:], np.zeros(coef[i, j, 1:].shape),
                                                               atol=precision)
                                    np.testing.assert_array_less(lower[i, j, 1:],
                                                                 np.zeros(lower[i, j, 1:].shape) + precision_int)
                                    np.testing.assert_array_less(np.zeros(lower[i, j, 1:].shape) - precision_int,
                                                                 upper[i, j, 1:])

                            est = LinearDML(model_y=LinearRegression(),
                                            model_t=LinearRegression(),
                                            linear_first_stages=False,
                                            featurizer=PolynomialFeatures(degree=1),
                                            fit_cate_intercept=False)
                            est.fit(y, T, X=X[:, :d_x], W=X[:, d_x:],
                                    inference=StatsModelsInference(cov_type='nonrobust'))
                            with pytest.raises(AttributeError) as e_info:
                                intercept = est.intercept_
                            with pytest.raises(AttributeError) as e_info:
                                intercept = est.intercept__interval(alpha=0.05)
                            coef = est.coef_.reshape(p, q, d_x + 1)
                            lower, upper = est.coef__interval(alpha=.001)
                            lower = lower.reshape(p, q, d_x + 1)
                            upper = upper.reshape(p, q, d_x + 1)
                            for i in range(p):
                                for j in range(q):
                                    np.testing.assert_allclose(coef[i, j, 0], 10 * i + j, rtol=0, atol=precision)
                                    np.testing.assert_array_less(lower[i, j, 0], 10 * i + j + precision_int)
                                    np.testing.assert_array_less(10 * i + j - precision_int, upper[i, j, 0])
                                    np.testing.assert_allclose(coef[i, j, 1], 1, atol=precision)
                                    np.testing.assert_array_less(lower[i, j, 1], 1)
                                    np.testing.assert_array_less(1, upper[i, j, 1])
                                    np.testing.assert_allclose(coef[i, j, 2:], np.zeros(coef[i, j, 2:].shape),
                                                               atol=precision)
                                    np.testing.assert_array_less(lower[i, j, 2:],
                                                                 np.zeros(lower[i, j, 2:].shape) + precision_int)
                                    np.testing.assert_array_less(np.zeros(lower[i, j, 2:].shape) - precision_int,
                                                                 upper[i, j, 2:])
                            XT = np.hstack([X, T])
                            (X1, X2, y1, y2,
                             X_final_first, X_final_sec, y_sum_first, y_sum_sec, n_sum_first, n_sum_sec,
                             var_first, var_sec) = _summarize(XT, y)
                            X = np.vstack([X1, X2])
                            y = np.concatenate((y1, y2))
                            X_final = np.vstack([X_final_first, X_final_sec])
                            y_sum = np.concatenate((y_sum_first, y_sum_sec))
                            n_sum = np.concatenate((n_sum_first, n_sum_sec))
                            var_sum = np.concatenate((var_first, var_sec))
                            first_half_sum = len(y_sum_first)

                            class SplitterSum:
                                def __init__(self):
                                    return

                                def split(self, X, T):
                                    return [(np.arange(0, first_half_sum), np.arange(first_half_sum, X.shape[0])),
                                            (np.arange(first_half_sum, X.shape[0]), np.arange(0, first_half_sum))]
                            est = LinearDML(
                                model_y=LinearRegression(),
                                model_t=LinearRegression(),
                                cv=SplitterSum(),
                                linear_first_stages=False,
                                discrete_treatment=False).fit(y_sum,
                                                              X_final[:, d:],
                                                              X=X_final[:, :d_x],
                                                              W=X_final[:, d_x:d],
                                                              freq_weight=n_sum,
                                                              sample_var=var_sum,
                                                              inference=StatsModelsInference(cov_type='nonrobust'))
                            intercept = est.intercept_.reshape((p, q))
                            lower_int, upper_int = est.intercept__interval(alpha=.001)
                            lower_int = lower_int.reshape((p, q))
                            upper_int = upper_int.reshape((p, q))
                            coef = est.coef_.reshape(p, q, d_x)
                            lower, upper = est.coef__interval(alpha=.001)
                            lower = lower.reshape(p, q, d_x)
                            upper = upper.reshape(p, q, d_x)
                            for i in range(p):
                                for j in range(q):
                                    np.testing.assert_allclose(intercept[i, j], 10 * i + j, rtol=0, atol=precision)
                                    np.testing.assert_array_less(lower_int[i, j], 10 * i + j + precision_int)
                                    np.testing.assert_array_less(10 * i + j - precision_int, upper_int[i, j])
                                    np.testing.assert_allclose(coef[i, j, 0], 1, atol=precision)
                                    np.testing.assert_array_less(lower[i, j, 0], 1)
                                    np.testing.assert_array_less(1, upper[i, j, 0])
                                    np.testing.assert_allclose(coef[i, j, 1:], np.zeros(coef[i, j, 1:].shape),
                                                               atol=precision)
                                    np.testing.assert_array_less(lower[i, j, 1:],
                                                                 np.zeros(lower[i, j, 1:].shape) + precision_int)
                                    np.testing.assert_array_less(np.zeros(lower[i, j, 1:].shape) - precision_int,
                                                                 upper[i, j, 1:])


class TestStatsModels2SLS(unittest.TestCase):
    def test_comp_with_IV2SLS(self):
        alpha = 0.05
        tol = 1e-2

        # dgp
        n = 1000
        Z = np.random.binomial(1, 0.5, size=(n, 1))
        T = 3.5 * Z + np.random.normal(0, 1, size=(n, 1))
        Y = 5.8 * T + np.random.normal(0, 1, size=(n, 1))

        # StatsModels2SLS
        est = StatsModels2SLS(cov_type="nonrobust")
        est.fit(Z, T, Y.ravel())

        # IV2SLS
        iv2sls = IV2SLS(Y.ravel(), T, Z).fit()

        assert est.cov_type == iv2sls.cov_type, "{}, {}".format(est.cov_type, iv2sls.cov_type)
        np.testing.assert_allclose(est.coef_, iv2sls.params, rtol=tol, atol=0)
        np.testing.assert_allclose(est._param_var, iv2sls.cov_params(), rtol=tol, atol=0)
        np.testing.assert_allclose(est.coef__interval(alpha=alpha)[
                                   0], iv2sls.conf_int(alpha=alpha)[:, 0], rtol=tol, atol=0)
        np.testing.assert_allclose(est.coef__interval(alpha=alpha)[
                                   1], iv2sls.conf_int(alpha=alpha)[:, 1], rtol=tol, atol=0)
