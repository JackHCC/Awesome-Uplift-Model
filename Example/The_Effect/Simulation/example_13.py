#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_13.py
@Author  :JackHCC
@Date    :2022/12/3 17:40 
@Desc    :

'''
import pandas as pd
import numpy as np
import seaborn as sns

# Get our data to bootstrap
iris = sns.load_dataset('iris')


def create_boot(d):
    N = d.shape[0]
    index = np.random.randint(0, N, size=N)
    d = d.iloc[index]
    return d


# Create a bootstrap sample
create_boot(iris)
