#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_1.py
@Author  :JackHCC
@Date    :2022/12/3 17:33 
@Desc    :

'''
import pandas as pd
import numpy as np

# If we want the results to be the same
# every time, set a seed
np.random.seed(1000)

# Create a DataFrame with pd.DataFrame. The size argument of the random
# functions gives us the number of observations
d = pd.DataFrame({
    # normal data is by default mean 0, sd 1
    'eps': np.random.normal(size=200),
    # Uniform data is by default from 0 to 1
    'Y': np.random.uniform(size=200),
    # We can use binomial to make binary data
    # with unequal probabilities
    'X': np.random.binomial(1, .2, size=200)
})
