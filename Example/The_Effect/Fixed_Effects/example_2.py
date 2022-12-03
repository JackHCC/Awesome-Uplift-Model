#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_2.py
@Author  :JackHCC
@Date    :2022/12/3 17:43 
@Desc    :

'''
import numpy as np
import statsmodels.formula.api as sm
from causaldata import gapminder

gm = gapminder.load_pandas().data
gm['logGDPpercap'] = gm['gdpPercap'].apply('log')

# Use C() to include binary variables for a categorical variable
m2 = sm.ols(formula='''lifeExp ~ logGDPpercap +
C(country)
''', data=gm).fit()
m2.summary()
