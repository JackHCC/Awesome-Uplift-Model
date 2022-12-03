#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_9.py
@Author  :JackHCC
@Date    :2022/12/3 17:38 
@Desc    :

'''
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

np.random.seed(1000)


# Data creation function. Let's also make the function more
# flexible - we can choose our own true effect!
def create_data(N, true):
    d = pd.DataFrame({'X': np.random.normal(size=N)})
    d['Y'] = true * d['X'] + np.random.normal(size=N)
    return d


# Estimation function. keep is the portion of data in each tail
# to keep. So .2 would keep the bottom and top 20% of X
def est_model(N, keep, true):
    d = create_data(N, true)

    # Agus' estimator!
    d = d.loc[(d['X'] <= np.quantile(d['X'], keep)) |
              (d['X'] >= np.quantile(d['X'], 1 - keep))]
    m = smf.ols('Y~X', data=d).fit()

    # Return the two things we want as an array
    ret = [m.params['X'], m.bse['X']]
    return ret


# Estimate the results 1000 times
results = [est_model(1000, .2, 2) for i in range(0, 1000)]

# Turn into a DataFrame
results = pd.DataFrame(results, columns=['coef', 'se'])

# Let's see what we got!
np.mean(results['coef'])
np.std(results['coef'])
np.mean(results['se'])
