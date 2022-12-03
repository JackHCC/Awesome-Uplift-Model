#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_3.py
@Author  :JackHCC
@Date    :2022/12/3 17:17 
@Desc    :

'''
import statsmodels.formula.api as sm
from causaldata import restaurant_inspections

df = restaurant_inspections.load_pandas().data

# We can simply add cov_type = 'HC3'
# to our fit!
m1 = sm.ols(formula='inspection_score ~ Year + Weekend',
            data=df).fit(cov_type='HC3')

m1.summary()
