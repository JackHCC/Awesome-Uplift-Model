#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_11.py
@Author  :JackHCC
@Date    :2022/12/3 17:39 
@Desc    :

'''
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

np.random.seed(1000)


# Have settings for strength of W -> X and for W -> Y
# These are relative to the standard deviation
# of the random components of X and Y, which are 1 each
# (np.random.normal() defaults to a standard deviation of 1)
def create_data(N, effectWX, effectWY):
    d = pd.DataFrame({'W': np.random.normal(size=N)})
    d['X'] = effectWX * d['W'] + np.random.normal(size=N)
    # True effect is 5
    d['Y'] = 5 * d['X'] + effectWY * d['W'] + np.random.normal(size=N)

    return d


# Our estimation function
def est_model(N, effectWX, effectWY):
    d = create_data(N, effectWX, effectWY)

    # Biased estimator - no W control!
    # But how bad is it?
    m = smf.ols('Y~X', data=d).fit()

    return (m.params['X'])


# Iteration function! Option iters determines number of iterations
def iterate(N, effectWX, effectWY, iters):
    results = [est_model(N, effectWX, effectWY) for i in range(0, iters)]

    # We want to know *how biased* it is, so compare to true-effect 5
    return (np.mean(results) - 5)


# Now try different settings to see how bias changes!
# Here we'll use a small number of iterations (200) to
# speed things up, but in general bigger is better

# Should be unbiased
iterate(2000, 0, 0, 200)
# Should still be unbiased
iterate(2000, 0, 1, 200)
# How much bias?
iterate(2000, 1, 1, 200)
# Now?
iterate(2000, .1, .1, 200)
# Does it make a difference whether the effect
# is stronger on X or Y?
iterate(2000, .5, .1, 200)
iterate(2000, .1, .5, 200)
