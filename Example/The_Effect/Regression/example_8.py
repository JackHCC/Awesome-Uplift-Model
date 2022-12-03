#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_8.py
@Author  :JackHCC
@Date    :2022/12/3 17:22 
@Desc    :

'''
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures
from causaldata import restaurant_inspections

df = restaurant_inspections.load_pandas().data

# Create a matrix with predictors, interactions, and squares
X = df[['NumberofLocations', 'Year', 'Weekend']]
X_all = PolynomialFeatures(2, interaction_only=False,
                           include_bias=False).fit_transform(X)

# Use LassoCV to pick a lambda with 20 chunks. normalize = True
# standardizes the variables. This particular model has trouble
# converging so let's give it more iterations to try with max_iter
reg = LassoCV(cv=20, normalize=True,
              max_iter=10000,
              ).fit(X_all, df['inspection_score'])

reg.coef_
# Looks like Weekend, squared Year, Year * NumberofLocations,
# and Year * Weekend all get dropped. So we can redo OLS without them

m1 = sm.ols(formula='''inspection_score ~
NumberofLocations + I(NumberofLocations ** 2) +
NumberofLocations: Weekend + Year
''', data=df).fit()
m1.summary()
