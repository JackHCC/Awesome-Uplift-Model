#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_7.py
@Author  :JackHCC
@Date    :2022/12/3 17:36 
@Desc    :

'''
import pandas as pd
import numpy as np
from itertools import product
from example_6 import create_clus_data
import statsmodels.formula.api as smf

np.random.seed(1000)


def est_model(N=200, T=10):
    # This uses create_clus_data from earlier
    d = create_clus_data(N, T)

    # Run a model that should be unbiased
    # if clustered errors themselves don't bias us!
    m = smf.ols('Y ~ X + W', data=d).fit()

    # Get the coefficient on X, which SHOULD be true value 3 on average
    x_coef = m.params['X']

    return x_coef


# Estimate our model!
est_model(200, 5)
