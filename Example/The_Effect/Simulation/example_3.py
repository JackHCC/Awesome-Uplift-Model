#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_3.py
@Author  :JackHCC
@Date    :2022/12/3 17:34 
@Desc    :

'''
import pandas as pd
import numpy as np

# Make sure the seed goes OUTSIDE the function. It makes the random data
# the same every time, but we want DIFFERENT results each time we run it
# (but the same set of different results, thus the seed)
np.random.seed(1000)


# Make a function with def. The "N = 200" argument gives it an argument N
# that we'll use for sample size. The "=200" sets the default to 200
def create_data(N=200):
    d = pd.DataFrame({
        'W': np.random.uniform(0, .1, size=N)})
    d['X'] = np.random.uniform(size=N) < .2 + d['W']
    d['Y'] = 3 * d['X'] + d['W'] + np.random.normal(size=N)
    # Use return() to send our created data back
    return d


# And run our function!
create_data(500)
