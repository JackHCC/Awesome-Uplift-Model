#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_4.py
@Author  :JackHCC
@Date    :2022/12/3 17:35 
@Desc    :

'''
import pandas as pd
import numpy as np
from itertools import product

np.random.seed(1000)


# N for number of individuals, T for time periods
def create_panel_data(N=200, T=10):
    # Use product() to get all combinations of individual and
    # time (if you want some to be incomplete, drop later)
    p = pd.DataFrame(
        product(range(0, N), range(0, T)))
    p.columns = ['ID', 't']

    # Individual- and time-varying variable
    p['W1'] = np.random.normal(size=N * T)

    # Individual data
    indiv_data = pd.DataFrame({
        'ID': range(0, N),
        'W2': np.random.normal(size=N)})

    # Bring them together
    p = p.merge(indiv_data, on='ID')

    # Create X, caused by W1 and W2
    p['X'] = 2 * p['W1'] + 1.5 * p['W2'] + np.random.normal(size=N * T)

    # And create Y. The true effect of X on Y is 3
    # But W1 and W2 have causal effects too
    p['Y'] = 3 * p['X'] + p['W1'] - 2 * p['W2'] + np.random.normal(size=N * T)
    return p


create_panel_data(100, 5)
