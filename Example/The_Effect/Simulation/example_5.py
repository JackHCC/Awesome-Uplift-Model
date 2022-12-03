#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_5.py
@Author  :JackHCC
@Date    :2022/12/3 17:35 
@Desc    :

'''
import pandas as pd
import numpy as np

np.random.seed(1000)


def create_het_data(N=200):
    d = pd.DataFrame({
        'X': np.random.uniform(size=N)})
    # Let the standard deviation of the error
    # Be related to X. Heteroskedasticity!
    d['Y'] = 3 * d['X'] + np.random.normal(scale=5 * d['X'])

    return d


create_het_data(500)
