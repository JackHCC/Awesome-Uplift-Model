#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_10.py
@Author  :JackHCC
@Date    :2022/12/3 17:38 
@Desc    :

'''
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from example_9 import create_data

np.random.seed(1000)


def est_model(N, keep, true):
    d = create_data(N, true)

    # Regular estimator
    m1 = smf.ols('Y~X', data=d).fit()

    # Agus' estimator!
    d = d.loc[(d['X'] <= np.quantile(d['X'], keep)) |
              (d['X'] >= np.quantile(d['X'], 1 - keep))]

    m2 = smf.ols('Y~X', data=d).fit()

    # Return the two things we want as an array
    ret = [m1.params['X'], m2.params['X']]

    return (ret)


# Estimate the results 1000 times
results = [est_model(1000, .2, 2) for i in range(0, 1000)]

# Turn into a DataFrame
results = pd.DataFrame(results, columns=['coef_reg', 'coef_agus'])

# Let's see what we got!
np.mean(results['coef_reg'])
np.std(results['coef_reg'])
np.mean(results['coef_agus'])
np.std(results['coef_agus'])
