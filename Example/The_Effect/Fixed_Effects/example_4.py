#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_4.py
@Author  :JackHCC
@Date    :2022/12/3 17:45 
@Desc    :

'''
import pandas as pd
import linearmodels as lm
from causaldata import gapminder

gm = gapminder.load_pandas().data
gm['logGDPpercap'] = gm['gdpPercap'].apply('log')

# Set our individual and time (index) for our data
gm = gm.set_index(['country', 'year'])

mod = lm.PanelOLS.from_formula(
    '''lifeExp ~ logGDPpercap +
EntityEffects
''', gm)

# Specify clustering when we fit the model
clfe = mod.fit(cov_type='clustered',
               cluster_entity=True)
print(clfe)
