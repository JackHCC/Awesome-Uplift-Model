# Pylift

> [Doc](https://pylift.readthedocs.io/en/latest/)

## Installation

**pylift** has only been tested on Python **3.6** and **3.7**. It currently requires the following package versions:

```
matplotlib >= 2.1.0
numpy >= 1.13.3
scikit-learn >= 0.19.1
scipy >= 1.0.0
xgboost >= 0.6a2
```

A `requirements.txt` file is included in the parent directory of the github repo that contains these lower-limit package versions, as these are the versions we have most extensively tested pylift on, but newer versions generally appear to work.

The package can be built from source (for the latest version) or simply sourced from pypi. To install from source, clone the repo and install, using the following commands:

```
git clone https://github.com/wayfair/pylift
cd pylift
pip install .
```

To upgrade, `git pull origin master` in the repo folder, and then run `pip install --upgrade --no-cache-dir .`.

Alternatively, install from pypi by simply running `pip install pylift`.



## ⭐Uplift

- 👉[Intro](https://pylift.readthedocs.io/en/latest/introduction.html)👈



## Quick start

To start, you simply need a `pandas.DataFrame` with a treatment column of 0s and 1s (0 for control, 1 for test) and a outcome column of 0s and 1s. Implementation can be as simple as follows:

```python
from pylift import TransformedOutcome
up = TransformedOutcome(df1, col_treatment='Treatment', col_outcome='Converted')

up.randomized_search()
up.fit(**up.rand_search_.best_params_)

up.plot(plot_type='aqini', show_theoretical_max=True)
print(up.test_results_.Q_aqini)
```

`up.fit()` can also be passed a flag `productionize=True`, which when `True` will create a productionizable model trained over the entire data set, stored in `self.model_final` (though it is contentious whether it’s safe to use a model that has not been evaluated in production – if you have enough data, it may be prudent not to). This can then be pickled with `self.model_final.to_pickle(PATH)`, as usually done with `sklearn`-style models.



## Usage: modeling

### Instantiation

If you followed the [Quick start](https://pylift.readthedocs.io/en/latest/evaluation), you hopefully already have a sense of how **pylift** is structured: the package is class-based and so the entire modeling process takes place within instantiation of a `TransformedOutcome` class. This method in particular implements the Transformed Outcome method, as described in [Introduction to uplift](https://pylift.readthedocs.io/en/latest/introduction).

In particular, the `TransformedOutcome` class inherits from a `BaseProxyMethod` class, and only adds to said class a `_transform_func` and an `_untransform_func` which perform the transformation to obtain Y∗Y∗ (the transformed outcome) from YY and WW (1 or 0 indicating the presence of a treatment) and vice versa, respectively. Custom transformation methods are therefore possible by explicitly providing the `transform_func` and `untransform_func` to `BaseProxyMethod`.

Instantiation is accomplished as follows:

```python
up = TransformedOutcome(df, col_treatment='Treatment', col_outcome='Converted')
```

A number of custom parameters can be passed, which are all documented in the docstring. Of particular note may be the `stratify` keyword argument (whose argument is directly passed to `sklearn.model_selection.train_test_split`).

The instantiation step accomplishes several things:

1. Define the transform function and transform the outcome (this is added to the dataframe you pass in, by default, as a new column, `TransformedOutcome`).
2. Split the data using `train_test_split`.
3. Set a random state (we like determinism!). This random state is used wherever possible.
4. Define an `untransform` function and use this to define a scoring function for hyperparameter tuning. The scoring function is saved within `up.randomized_search_params` and `up.grid_search_params`, which are dictionaries that are used by default whenever `up.randomized_search()` or `up.grid_search()` are called.
5. Define some default hyperparameters.

If you’d alternatively like to specify your own train-test split, a tuple of dataframes can also be passed in place of `df`, with the training data as the first element and the test data as the second (e.g. `TransformedOutcome((df_train, df_test), col_treatment...)`.

### Fit and hyperparameter tunings: passing custom parameters

Anything that can be taken by `RandomizedSearchCV()`, `GridSearchCV()`, or `Regressor()` can be similarly passed to `up.randomized_search`, `up.grid_search`, or `up.fit`, respectively.

```python
up.fit(max_depth=2, nthread=-1)
```

`XGBRegressor` is the default regressor, but a different `Regressor` object can also be used. To do this, pass the object to the keyword argument `sklearn_model` during `TransformedOutcome` instantiation.

```python
up = TransformedOutcome(df, col_treatment='Test', col_outcome='Converted', sklearn_model=RandomForestRegressor)

grid_search_params = {
    'estimator': RandomForestRegressor(),
    'param_grid': {'min_samples_split': [2,3,5,10,30,100,300,1000,3000,10000]},
    'verbose': True,
    'n_jobs': 35,
}
up.grid_search(**grid_search_params)
```

We tend to prefer `xgboost`, however, as it tends to give favorable results quickly, while also allowing the option for a custom objective function. This extensibility allows for the possibility of an objective function that takes into account P(W=1)P(W=1) within each leaf, though we because we have had mixed results with this approach, we have left the package defaults as is.

Regardless of what regressor you use, the `RandomizedSearchCV` default params are contained in `up.randomized_search_params`, and the `GridSearchCV` params are located in `up.grid_search_params`. These can be manually replaced, but doing so will remove the scoring functions, so it is highly recommended that any alterations to these class attributes be done as an update, or that alterations be simply passed as arguments to `randomized_search` or `grid_search`, as shown above.

Moreover, any time an argument is passed to `up.randomized_search()`, it is saved to `up.randomized_search_params`. This is as intended (though may be changed in a future iteration), but a small side effect is that if an invalid argument is passed to `up.randomized_search()`, future calls to `up.randomized_search()` will fail until the dictionary is repaired.

### Accessing sklearn objects

The class objects produced by the sklearn classes, `RandomizedSearchCV`, `GridSearchCV`, `XGBRegressor`, etc. are preserved in the `TransformedOutcome` class as class attributes.

```
up.randomized_search` -> `up.rand_search_
up.grid_search` -> `up.grid_search_
up.fit` -> `up.model
up.fit(productionize=True)` -> `up.model_final
```

### Continuous outcome

The `TransformedOutcome` class natively supports continuous outcomes as well, though the theoretical max and practical max curves in `UpliftEval` are not as well defined, so will not be correct. Negative values, however, are not permitted. Care should be taken, however, when dealing with continuous outcomes with anomalously long-tailed distributions, as the tail will often dominate the split choices, leading to an overfit model. It is often prudent to carefully choose hyperparameters, winsorize, or even transform the positive values to curtail this effect.



## Usage: EDA

Once a `TransformedOutcome()` object has been instantiated, **pylift** offers a couple methods for simple feature EDA: `up.NIV()` (Net Information Value) and `up.NWOE()` (Net Weight of Evidence).

See [Data Exploration with Weight of Evidence and Information Value in R](https://multithreaded.stitchfix.com/blog/2015/08/13/weight-of-evidence/) for more details.

### Net Weight of Evidence

*Weight of Evidence* comes from a simple Bayesian decomposition of relative lift as a function of features:

$$logP(Y=1|Xj)/P(Y=0|Xj)=logP(Y=1)/P(Y=0)+logP(Xj|Y=1)/P(Xj|Y=0)$$

Net Weight of Evidence (NWOE) is the difference in Weight of Evidence (WOE) between the treatment and control groups.

$$NWOE=WOEt−WOEc$$

Where Weight of Evidence is defined as:

$$WOEij=logP(Xj\elemBi|Y=1)P(Xj\elemBi|Y=0)$$

where BiBi indicates a bin ii, and the subscript jj indicates a particular feature.

This can be accessed in **pylift** as follows:

```python
up = TransformedOutcome(df)
up.NWOE()
```

`up.NWOE()` can take two arguments: `n_bins` (the number of bins) and `feats_to_use` (a subset of features over which to calculate NWOE).

The base routine can be accessed from the `pylift.explore` module (`from pylift.explore.base import _NWOE`).

### Net Information Value

*Information Value* is the sum of all WOE values, weighted by the absolute difference in the numerator and denominator.

$$IVj=∫logP(Xj|Y=1)/P(Xj|Y=0)(P(Xj|Y=1)−P(Xj|Y=0)) dx$$

For the *net* weight of evidence, the numerator and denominator can be rewritten as:

$$NWOE_j=logP(xj|Y=1,W=1)P(xj|Y=0,W=0)/(P(xj|Y=1,W=0)P(xj|Y=0,W=1))$$

Net information value, then, sums the NWOE values, weighted by the difference between the above numerator and denominator.

This can be accessed in **pylift** as follows:

```python
up = TransformedOutcome(df)
up.NIV()
```

`up.NIV()` accepts `feats_to_use` and `n_bins`, as it requires NWOE to be calculated as a pre-requisite. But `up.NIV()` also accepts `n_iter` – `up.NIV()` will bootstrap the training set to obtain error bars, and `n_iter` specifies the number of iterations to use.

The base routine can be accessed from the `pylift.explore` module (`from pylift.explore.base import _NIV`).



## Usage: evaluation

- [Here](https://pylift.readthedocs.io/en/latest/evaluation.html)

```python
from pylift.eval import UpliftEval
upev = UpliftEval(treatment, outcome, predictions)
upev.plot(plot_type='aqini')
```

