#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_3.py
@Author  :JackHCC
@Date    :2022/12/3 17:27 
@Desc    :

'''
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
# The more-popular matching tools in sklearn
# are more geared towards machine learning than statistical inference
from causalinference.causal import CausalModel
from causaldata import black_politicians

br = black_politicians.load_pandas().data

# Get our outcome, treatment, and matching variables as numpy arrays
Y = br['responded'].to_numpy()
D = br['leg_black'].to_numpy()
X = br[['medianhhincom', 'blackpercent', 'leg_democrat']].to_numpy()

# Set up our model
M = CausalModel(Y, D, X)

# Estimate the propensity score using logit
M.est_propensity()

# Trim the score with improved algorithm trim_s to improve balance
M.trim_s()

# If we want to use the scores elsewhere, export them
# (we could have also done this with sm.Logit)
br['ps'] = M.propensity['fitted']

# We can estimate the effect directly (note this uses "doubly robust" methods
# as will be later described, which is why it doesn't match the sm.wls result)
M.est_via_weighting()

print(M.estimates)

# Or we can do our own weighting
br['ipw'] = br['leg_black'] * (1 / br['ps']
                               ) + (1 - br['leg_black']) * (1 / (1 - br['ps']))

# Now, use the weights to estimate the effect (this will produce
# incorrect standard errors unless we bootstrap the whole process,
# as in the doubly robust section later, or the Simulation chapter)
m = sm.wls(formula='responded ~ leg_black',
           weights=br['ipw'], data=br).fit()

m.summary()
