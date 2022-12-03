#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_6.py
@Author  :JackHCC
@Date    :2022/12/3 17:36 
@Desc    :

'''
import pandas as pd
import numpy as np
from itertools import product

np.random.seed(1000)


# N for number of individuals, T for time periods
def create_clus_data(N=200, T=10):
    # We're going to create errors clustered at the
    # ID level. So we can follow our steps from making panel data
    p = pd.DataFrame(
        product(range(0, N), range(0, T)))
    p.columns = ['ID', 't']

    # Individual- and time-varying variable
    p['W'] = np.random.normal(size=N * T)

    # Now an individual-specific error cluster
    indiv_data = pd.DataFrame({
        'ID': range(0, N),
        'C': np.random.normal(size=N)})

    # Bring them together
    p = p.merge(indiv_data, on='ID')

    # Create X
    p['X'] = 2 * p['W'] + np.random.normal(size=N * T)

    # And create Y. The error term has two components: the individual
    # cluster C, and the  individual-and-time-varying element
    p['Y'] = 3 * p['X'] + (p['C'] + np.random.normal(size=N * T))
    return p


create_clus_data(100, 5)
