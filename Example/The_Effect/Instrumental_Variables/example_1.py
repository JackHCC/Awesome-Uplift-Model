#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_1.py
@Author  :JackHCC
@Date    :2022/12/3 17:52 
@Desc    :

'''
import pandas as pd
from linearmodels.iv import IV2SLS
from causaldata import social_insure

d = social_insure.load_pandas().data

# Create an control-variable DataFrame
# including dummies for village
controls = pd.concat([d[['male', 'age', 'agpop', 'ricearea_2010',
                         'literacy', 'intensive', 'risk_averse', 'disaster_prob']],
                      pd.get_dummies(d[['village']])],
                     axis=1)

# Create model and fit separately
# since we want to cluster, and will use
# m.notnull to see which observations to drop
m = IV2SLS(d['takeup_survey'],
           controls,
           d['pre_takeup_rate'],
           d['default'])
second_stage = m.fit(cov_type='clustered',
                     clusters=d['address'][m.notnull])

# If we want the first stage we must do it ourselves!
first_stage = IV2SLS(d['pre_takeup_rate'],
                     pd.concat([controls, d['default']], axis=1), None,
                     None).fit(cov_type='clustered',
                               clusters=d['address'][m.notnull])

print(first_stage)
print(second_stage)
