#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_6.py
@Author  :JackHCC
@Date    :2022/12/3 17:21 
@Desc    :

'''
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from causaldata import restaurant_inspections

df = restaurant_inspections.load_pandas().data


# There are bootstrap estimators in Python but generally not designed for OLS
# So we'll make our own. Also see the Simulation chapter.
def our_reg(DF):
    # Resample with replacement
    resampler = np.random.randint(0, len(df), len(df))
    DF = DF.iloc[resampler]

    # Run our estimate
    m = sm.ols(formula='inspection_score ~ Year',
               data=DF).fit()
    # Get all coefficients
    return (dict(m.params))

    # Run the function 2000 times and store results
    results = [our_reg(df) for i in range(0, 2000)]

    # Turn results into a data frame
    results = pd.DataFrame(results)

    # Mean and standard deviation are estimate and SE
    results.describe()


our_reg(df)
