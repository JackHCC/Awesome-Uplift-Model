#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_4.py
@Author  :JackHCC
@Date    :2022/12/3 17:28 
@Desc    :

'''
import pandas as pd
import numpy as np
import seaborn as sns
from causaldata import black_politicians

br = black_politicians.load_pandas().data

# Overlaid treatment/control density in raw data
fig1 = sns.kdeplot(data=br,
                   x='medianhhincom',
                   hue='leg_black',
                   common_norm=False)
fig1.plot()

# Start the new plot
fig1.get_figure().clf()

# Add weights from any weighting method to check post-matching density
# Here we have the ipw variable from our propensity score matching
# in previous code blocks (make sure you run those first!)
fig2 = sns.kdeplot(data=br, x='medianhhincom',
                   hue='leg_black', common_norm=False, weights='ipw')
fig2.plot()
