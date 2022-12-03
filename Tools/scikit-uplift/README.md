# Scikit-Uplift

> [Doc](https://www.uplift-modeling.com/en/latest/index.html)

## Installation

**Install** the package by the following command from [PyPI](https://pypi.org/project/scikit-uplift/):

```
pip install scikit-uplift
```

Or install from [source](https://github.com/maks-sh/scikit-uplift):

```
git clone https://github.com/maks-sh/scikit-uplift.git
cd scikit-uplift
python setup.py install
```



## Quick Start

### Train and predict your uplift model

Use the intuitive python API to train uplift models with [sklift.models](https://www.uplift-modeling.com/en/latest/api/models/index.html).

```python
# import approaches
from sklift.models import SoloModel, ClassTransformation
# import any estimator adheres to scikit-learn conventions.
from lightgbm import LGBMClassifier

# define models
estimator = LGBMClassifier(n_estimators=10)

# define metamodel
slearner = SoloModel(estimator=estimator)

# fit model
slearner.fit(
    X=X_tr,
    y=y_tr,
    treatment=trmnt_tr,
)

# predict uplift
uplift_slearner = slearner.predict(X_val)
```

### Evaluate your uplift model

Uplift model evaluation metrics are available in [sklift.metrics](https://www.uplift-modeling.com/en/latest/api/metrics/index.html).

```python
# import metrics to evaluate your model
from sklift.metrics import (
    uplift_at_k, uplift_auc_score, qini_auc_score, weighted_average_uplift
)


# Uplift@30%
uplift_at_k = uplift_at_k(y_true=y_val, uplift=uplift_slearner,
                          treatment=trmnt_val,
                          strategy='overall', k=0.3)

# Area Under Qini Curve
qini_coef = qini_auc_score(y_true=y_val, uplift=uplift_slearner,
                           treatment=trmnt_val)

# Area Under Uplift Curve
uplift_auc = uplift_auc_score(y_true=y_val, uplift=uplift_slearner,
                              treatment=trmnt_val)

# Weighted average uplift
wau = weighted_average_uplift(y_true=y_val, uplift=uplift_slearner,
                              treatment=trmnt_val)
```

### Vizualize the results

Visualize performance metrics with [sklift.viz](https://www.uplift-modeling.com/en/latest/api/viz/index.html).

```python
from sklift.viz import plot_qini_curve
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)
ax.set_title('Qini curves')

plot_qini_curve(
    y_test, uplift_slearner, trmnt_test,
    perfect=True, name='Slearner', ax=ax
);

plot_qini_curve(
    y_test, uplift_revert, trmnt_test,
    perfect=False, name='Revert label', ax=ax
);
```

```python
from sklift.viz import plot_uplift_curve
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)
ax.set_title('Uplift curves')

plot_uplift_curve(
    y_test, uplift_slearner, trmnt_test,
    perfect=True, name='Slearner', ax=ax
);

plot_uplift_curve(
    y_test, uplift_revert, trmnt_test,
    perfect=False, name='Revert label', ax=ax
);
```

```python
from sklift.viz import plot_uplift_by_percentile

plot_uplift_by_percentile(y_true=y_val, uplift=uplift_preds,
                          treatment=treat_val, kind='bar')
```



## ⭐User Guide

- 👉[Here](https://www.uplift-modeling.com/en/latest/user_guide/index.html)👈



## API sklift

This is the modules reference of scikit-uplift.

- sklift.models
  - [sklift.models.SoloModel](https://www.uplift-modeling.com/en/latest/api/models/SoloModel.html)
  - [sklift.models.ClassTransformation](https://www.uplift-modeling.com/en/latest/api/models/ClassTransformation.html)
  - [sklift.models.ClassTransformationReg](https://www.uplift-modeling.com/en/latest/api/models/ClassTransformationReg.html)
  - [sklift.models.TwoModels](https://www.uplift-modeling.com/en/latest/api/models/TwoModels.html)
- sklift.metrics
  - [sklift.metrics.uplift_at_k](https://www.uplift-modeling.com/en/latest/api/metrics/uplift_at_k.html)
  - [sklift.metrics.uplift_curve](https://www.uplift-modeling.com/en/latest/api/metrics/uplift_curve.html)
  - [sklift.metrics.perfect_uplift_curve](https://www.uplift-modeling.com/en/latest/api/metrics/perfect_uplift_curve.html)
  - [sklift.metrics.uplift_auc_score](https://www.uplift-modeling.com/en/latest/api/metrics/uplift_auc_score.html)
  - [sklift.metrics.qini_curve](https://www.uplift-modeling.com/en/latest/api/metrics/qini_curve.html)
  - [sklift.metrics.perfect_qini_curve](https://www.uplift-modeling.com/en/latest/api/metrics/perfect_qini_curve.html)
  - [sklift.metrics.qini_auc_score](https://www.uplift-modeling.com/en/latest/api/metrics/qini_auc_score.html)
  - [sklift.metrics.weighted_average_uplift](https://www.uplift-modeling.com/en/latest/api/metrics/weighted_average_uplift.html)
  - [sklift.metrics.uplift_by_percentile](https://www.uplift-modeling.com/en/latest/api/metrics/uplift_by_percentile.html)
  - [sklift.metrics.response_rate_by_percentile](https://www.uplift-modeling.com/en/latest/api/metrics/response_rate_by_percentile.html)
  - [sklift.metrics.treatment_balance_curve](https://www.uplift-modeling.com/en/latest/api/metrics/treatment_balance_curve.html)
  - [sklift.metrics.average_squared_deviation](https://www.uplift-modeling.com/en/latest/api/metrics/average_squared_deviation.html)
  - [sklift.metrics.max_prof_uplift](https://www.uplift-modeling.com/en/latest/api/metrics/max_prof_uplift.html)
  - [sklift.metrics.make_uplift_scorer](https://www.uplift-modeling.com/en/latest/api/metrics/make_uplift_scorer.html)
- sklift.viz
  - [sklift.viz.plot_uplift_preds](https://www.uplift-modeling.com/en/latest/api/viz/plot_uplift_preds.html)
  - [sklift.viz.plot_qini_curve](https://www.uplift-modeling.com/en/latest/api/viz/plot_qini_curve.html)
  - [sklift.viz.plot_uplift_curve](https://www.uplift-modeling.com/en/latest/api/viz/plot_uplift_curve.html)
  - [sklift.viz.plot_treatment_balance_curve](https://www.uplift-modeling.com/en/latest/api/viz/plot_treatment_balance_curve.html)
  - [sklift.viz.plot_uplift_by_percentile](https://www.uplift-modeling.com/en/latest/api/viz/plot_uplift_by_percentile.html)
- sklift.datasets
  - [sklift.datasets.clear_data_dir](https://www.uplift-modeling.com/en/latest/api/datasets/clear_data_dir.html)
  - [sklift.datasets.get_data_dir](https://www.uplift-modeling.com/en/latest/api/datasets/get_data_dir.html)
  - sklift.datasets.fetch_lenta
    - [Lenta Uplift Modeling Dataset](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_lenta.html#lenta-uplift-modeling-dataset)
  - sklift.datasets.fetch_x5
    - [X5 RetailHero Uplift Modeling Dataset](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_x5.html#x5-retailhero-uplift-modeling-dataset)
  - sklift.datasets.fetch_criteo
    - [Criteo Uplift Modeling Dataset](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_criteo.html#criteo-uplift-modeling-dataset)
  - sklift.datasets.fetch_hillstrom
    - [Kevin Hillstrom Dataset: MineThatData](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_hillstrom.html#kevin-hillstrom-dataset-minethatdata)
  - sklift.datasets.fetch_megafon
    - [MegaFon Uplift Competition Dataset](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_megafon.html#megafon-uplift-competition-dataset)
