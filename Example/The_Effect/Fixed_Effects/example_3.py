#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_3.py
@Author  :JackHCC
@Date    :2022/12/3 17:45 
@Desc    :

'''
import linearmodels as lm
from causaldata import gapminder

gm = gapminder.load_pandas().data
gm['logGDPpercap'] = gm['gdpPercap'].apply('log')

# Set our individual and time (index) for our data
gm = gm.set_index(['country', 'year'])

# Specify the regression model
# And estimate with both sets of fixed effects
# EntityEffects and TimeEffects
# (this function can't handle more than two)
mod = lm.PanelOLS.from_formula(
    '''lifeExp ~ logGDPpercap +
EntityEffects +
TimeEffects
''', gm)

twfe = mod.fit()
print(twfe)
