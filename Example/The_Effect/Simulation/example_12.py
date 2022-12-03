#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_12.py
@Author  :JackHCC
@Date    :2022/12/3 17:40 
@Desc    :

'''
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

np.random.seed(1000)


# Follow the description in the text for data creation. Since we want to
# get minimum sample size and minimum detectable effect, allow both sample
# size and effect to vary.
# diff is the difference in effects between boys and girls
def create_data(N, effect, diff):
    d = pd.DataFrame({'W': np.random.normal(size=N),
                      'girl': np.random.randint(2, size=N)})
    # A one-SD change in W makes treatment 10% less likely
    d['Training'] = np.random.uniform(size=N) + .1 * d['W'] < .5
    d['Test'] = effect * d['Training'] + diff * d['girl'] * d['Training']
    d['Test'] = d['Test'] + 4 * d['W'] + np.random.normal(scale=9, size=N)
    return (d)


# Our estimation function
def est_model(N, effect, diff):
    d = create_data(N, effect, diff)

    # Our model
    m = smf.ols('Test~girl*Training + W', data=d).fit()

    # By looking we can spot that the name of the
    # interaction term is girl:Training[T.True]
    sig = m.pvalues['girl:Training[T.True]'] < .05
    return (sig)


# Iteration function!
def iterate(N, effect, diff, iters):
    results = [est_model(N, effect, diff) for i in range(0, iters)]

    # We want to know statistical power,
    # i.e., the proportion of significant results
    return (np.mean(results))


# Let's find the minimum sample size
mss = [[N, iterate(N, 2, .8, 500)] for
       N in [10000, 15000, 20000, 25000]]
# Look for the first N with power above 90%
pd.DataFrame(mss, columns=['N', 'Power'])

# Now for the minimum detectable effect
mde = [[diff, iterate(2000, 2, diff, 500)] for
       diff in [.8, 1.6, 2.4, 3.2]]
pd.DataFrame(mde, columns=['Effect', 'Power'])
