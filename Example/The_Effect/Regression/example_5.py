#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_5.py
@Author  :JackHCC
@Date    :2022/12/3 17:20 
@Desc    :

'''
import statsmodels.formula.api as sm
from causaldata import restaurant_inspections

df = restaurant_inspections.load_pandas().data

# We can simply add cov_type = 'cluster' to our fit, and specify the groups
m1 = sm.ols(formula='inspection_score ~ Year',
            data=df).fit(cov_type='cluster',
                         cov_kwds={'groups': df['Weekend']})

m1.summary()
