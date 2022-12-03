#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_4.py
@Author  :JackHCC
@Date    :2022/12/3 17:17 
@Desc    :

'''

import statsmodels.formula.api as sm
from causaldata import restaurant_inspections

df = restaurant_inspections.load_pandas().data

# Get our data into a single time series!
df = df.groupby('Year').agg([('mean')])

# Only use the years without a gap
df = df.query('Year <= 2009')

# We can simply add cov_type = 'HAC' to our fit, with maxlags specified
m1 = sm.ols(formula='inspection_score ~ Weekend',
            data=df).fit(cov_type='HAC',
                         cov_kwds={'maxlags': 1})

m1.summary()
# Note that Python uses a "classic" form of HAC that does not apply
# "pre-whiting", so results are often different from other languages
