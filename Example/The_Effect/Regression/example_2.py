#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_2.py
@Author  :JackHCC
@Date    :2022/12/3 17:16 
@Desc    :

'''
from stargazer.stargazer import Stargazer
import statsmodels.formula.api as sm
from causaldata import restaurant_inspections

df = restaurant_inspections.load_pandas().data

# sm.logit wants the dependent variable to be numeric
df['Weekend'] = 1 * df['Weekend']

# Use sm.logit to run logit
m1 = sm.logit(formula="Weekend ~ Year", data=df).fit()

# See the result
# m1.summary() would also work
Stargazer([m1])

# And get marginal effects
m1.get_margeff().summary()
