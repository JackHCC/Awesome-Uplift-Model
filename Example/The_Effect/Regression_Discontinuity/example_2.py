#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_2.py
@Author  :JackHCC
@Date    :2022/12/3 17:55 
@Desc    :

'''
import numpy as np
import statsmodels.formula.api as smf
from causaldata import gov_transfers

d = gov_transfers.load_pandas().data

# Run the polynomial model
m1 = smf.ols('''Support~Income_Centered*Participation + 
I(Income_Centered**2)*Participation''', d).fit()


# Create the kernel function
def kernel(x):
    # To start at a weight of 0 at x = 0,
    # and impose a bandwidth of .01, we need a "slope" of -1/.01 = 100
    # and to go in either direction use the absolute value
    w = 1 - 100 * np.abs(x)
    # if further away than .01, the weight is 0, not negative
    w = np.maximum(0, w)
    return w


# Run the linear model with weights using wls
m2 = smf.wls('Support~Income_Centered*Participation', d,
             weights=kernel(d['Income_Centered'])).fit()

m1.summary()
m2.summary()
