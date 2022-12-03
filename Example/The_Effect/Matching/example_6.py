#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_6.py
@Author  :JackHCC
@Date    :2022/12/3 17:31 
@Desc    :

'''
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from causaldata import black_politicians

br = black_politicians.load_pandas().data


# As mentioned, est_via_weighting from causalinfernece already does
# doubly robust estimation but it's a different kind!
# Let's do Wooldridge here.
def ipwra(br):
    # Estimate propensity
    m = sm.logit('leg_black ~ medianhhincom + blackpercent + leg_democrat',
                 data=br)
    # Get fitted values and turn them into probabilities
    m = m.fit().fittedvalues
    br['ps'] = np.exp(m) / (1 + np.exp(m))

    # Trim observations outside of treated range
    minrange = np.min(br.loc[br['leg_black'] == 1, 'ps'])
    maxrange = np.max(br.loc[br['leg_black'] == 1, 'ps'])
    br = br.loc[(br['ps'] >= minrange) & (br['ps'] <= maxrange)]

    # Get inverse probability score
    br['ipw'] = br['leg_black'] * (1 / br['ps']
                                   ) + (1 - br['leg_black']) * (1 / (1 - br['ps']))

    # Regress treated and nontreated separately,
    # then predict for whole sample
    mtreat = sm.wls('''responded ~ medianhhincom + 
    blackpercent + leg_democrat''',
                    weights=br.loc[br['leg_black'] == 1, 'ipw'],
                    data=br.loc[br['leg_black'] == 1])
    mcontrol = sm.ols('''responded ~ medianhhincom + 
    blackpercent + leg_democrat''',
                      weights=br.loc[br['leg_black'] == 0, 'ipw'],
                      data=br.loc[br['leg_black'] == 0])

    treat_predict = mtreat.fit().predict(exog=br)
    con_predict = mcontrol.fit().predict(exog=br)

    # Compare means
    diff = np.mean(treat_predict) - np.mean(con_predict)
    return diff


# And a wrapper function to bootstrap
def ipwra_b(br):
    n = br.shape[0]
    br = br.iloc[np.random.randint(n, size=n)]
    diff = ipwra(br)
    return diff


# Run once on the original data to get our estimate
est = ipwra(br)

# And then a bunch of times to get the sampling distribution
dist = [ipwra_b(br) for i in range(0, 2000)]

# Our estimate
est
# and its standard error
np.std(dist)
