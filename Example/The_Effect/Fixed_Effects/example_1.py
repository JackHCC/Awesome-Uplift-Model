#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_1.py
@Author  :JackHCC
@Date    :2022/12/3 17:43 
@Desc    :

'''
import numpy as np
import statsmodels.formula.api as sm
from causaldata import gapminder

gm = gapminder.load_pandas().data

# Put GDP per capita in log format since it's very skewed
gm['logGDPpercap'] = gm['gdpPercap'].apply('log')

# Use groupby to perform calculations by group
# Then use transform to subtract each variable's
# within-group mean to get within variation
gm[['logGDPpercap_within', 'lifeExp_within']] = (gm.
                                                 groupby('country')[['logGDPpercap', 'lifeExp']].
                                                 transform(lambda x: x - np.mean(x)))

# Analyze the within variation
m1 = sm.ols(formula='lifeExp_within ~ logGDPpercap_within',
            data=gm).fit()
m1.summary()
