#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_8.py
@Author  :JackHCC
@Date    :2022/12/3 17:36 
@Desc    :

'''
import pandas as pd
import numpy as np
from itertools import product
import statsmodels.formula.api as smf
from example_7 import est_model

np.random.seed(1000)

# This runs est_model once for each iteration as it iterates through
# the range from 0 to 999 (1000 times total)
estimates = [est_model(200, 5) for i in range(0, 1000)]
