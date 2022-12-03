#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_1.py
@Author  :JackHCC
@Date    :2022/12/3 17:23 
@Desc    :

'''
import pandas as pd
import numpy as np
# The more-popular matching tools in sklearn
# are more geared towards machine learning than statistical inference
from causalinference.causal import CausalModel
from causaldata import black_politicians

br = black_politicians.load_pandas().data

# Get our outcome, treatment, and matching variables
# We need these as numpy arrays
Y = br['responded'].to_numpy()
D = br['leg_black'].to_numpy()
X = br[['medianhhincom', 'blackpercent', 'leg_democrat']].to_numpy()

# Set up our model
M = CausalModel(Y, D, X)

# Fit, using Mahalanobis distance
M.est_via_matching(weights='maha', matches=1)

print(M.estimates)
# Note it automatically calcultes average treatments on
# average, on treated, and on untreated/control (ATC)
