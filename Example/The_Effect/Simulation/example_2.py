#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_2.py
@Author  :JackHCC
@Date    :2022/12/3 17:34 
@Desc    :

'''
import pandas as pd
import numpy as np

# If we want the results to be the same every time, set a seed
np.random.seed(1000)

# Create a DataFrame with pd.DataFrame. The size argument of the random
# functions gives us the number of observations
d = pd.DataFrame({
    # Have W go from 0 to .1
    'W': np.random.uniform(0, .1, size=200)})

# Higher W makes X = 1 more likely
d['X'] = np.random.uniform(size=200) < .2 + d['W']

# The true effect of X on Y is 3
d['Y'] = 3 * d['X'] + d['W'] + np.random.normal(size=200)
