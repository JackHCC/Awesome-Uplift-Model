#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_1.py
@Author  :JackHCC
@Date    :2022/12/3 17:09 
@Desc    :

'''

import pandas as pd
import statsmodels.formula.api as sm
from stargazer.stargazer import Stargazer
from causaldata import restaurant_inspections

res = restaurant_inspections.load_pandas().data

# Perform the first, one-predictor regression
# use the sm.ols() function, with ~ telling us what
# the dependent variable varies over
m1 = sm.ols(formula='inspection_score ~ NumberofLocations',
            data=res).fit()

# Now add year as a control
# Just use + to add more terms to the regression
m2 = sm.ols(formula='inspection_score ~ NumberofLocations + Year',
            data=res).fit()

# Open a file to write to
f = open('regression_table.html', 'w')

# Give Stargazer a list of the models we want in our table
# and save to file
regtable = Stargazer([m1, m2])
f.write(regtable.render_html())
f.close()
