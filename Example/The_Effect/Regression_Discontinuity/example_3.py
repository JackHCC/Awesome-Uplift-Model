#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_3.py
@Author  :JackHCC
@Date    :2022/12/3 17:55 
@Desc    :

'''
import pandas as pd
from linearmodels.iv import IV2SLS
from causaldata import mortgages

d = mortgages.load_pandas().data

# Create an above variable as an instrument
d['above'] = d['qob_minus_kw'] > 0
# Apply a bandwidth of 12 quarters on either side
d = d.query('abs(qob_minus_kw) < 12')

# Create an control-variable DataFrame
# including dummies for bpl and qob
controls = pd.concat([d[['nonwhite']],
                      pd.get_dummies(d[['bpl']])], axis=1)

d['qob'] = pd.Categorical(d['qob'])
# Drop one since we already have full rank with bpl
# (we'd also drop_first with bpl if we did add_constant)
controls = pd.concat([controls,
                      pd.get_dummies(d[['qob']],
                                     drop_first=True)], axis=1)
# the RDD terms:
# qob_minus_kw by itself is a control
controls = pd.concat([controls, d[['qob_minus_kw']]], axis=1)

# we need interactions for the second stage
d['interaction_vet'] = d['vet_wwko'] * d['qob_minus_kw']
# and the first
d['interaction_above'] = d['above'] * d['qob_minus_kw']

# Now we estimate!
m = IV2SLS(d['home_ownership'], controls,
           d[['vet_wwko', 'interaction_vet']],
           d[['above', 'interaction_above']])

# With robust standard errors
m.fit(cov_type='robust')
