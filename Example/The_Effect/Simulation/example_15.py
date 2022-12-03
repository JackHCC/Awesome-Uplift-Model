#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_15.py
@Author  :JackHCC
@Date    :2022/12/3 17:42 
@Desc    :

'''
import pandas as pd
import numpy as np

np.random.seed(100)

# Example data
d = pd.DataFrame({'ID': [1, 1, 2, 2, 3, 3],
                  'X': [1, 2, 3, 4, 5, 6]})
# Now, get our data frame just of IDs
IDs = pd.DataFrame({'ID': np.unique(d['ID'])})

# Our bootstrap resampling function


def create_boot(d, IDs):
    # Resample our ID data
    N = IDs.shape[0]
    index = np.random.randint(0, N, size=N)
    bs_ID = IDs.iloc[index]

    # And our full data
    bs_d = d.merge(bs_ID, how='inner', on='ID')
    return (bs_d)


# Create a cluster bootstrap data set
create_boot(d, IDs)
