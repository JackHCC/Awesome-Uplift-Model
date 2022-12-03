#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_5.py
@Author  :JackHCC
@Date    :2022/12/3 17:31 
@Desc    :

'''
import pandas as pd
import statsmodels.formula.api as sm
from causaldata import black_politicians

br = black_politicians.load_pandas().data

# This copies the CEM code from the CEM section
# See that section's code for comments and notes
br['inc_bins'] = pd.qcut(br['medianhhincom'], 6)
br['bp_bins'] = pd.qcut(br['blackpercent'], 6)
treated = br.loc[br['leg_black'] == 1
                 ].groupby(['inc_bins', 'bp_bins', 'leg_democrat']
                           ).size().to_frame('treated')
control = br.loc[br['leg_black'] == 0
                 ].groupby(['inc_bins', 'bp_bins', 'leg_democrat']
                           ).size().to_frame('control')

# Merge the counts back in
br = br.join(treated, on=['inc_bins', 'bp_bins', 'leg_democrat'])
br = br.join(control, on=['inc_bins', 'bp_bins', 'leg_democrat'])

# Create weights
br['weight'] = 0
br.loc[(br['leg_black'] == 1) & (br['control'] > 0), 'weight'] = 1
totalcontrols = sum(br.loc[br['leg_black'] == 0]['treated'] > 0)
totaltreated = sum(br.loc[br['leg_black'] == 1]['control'] > 0)
br['controlweights'] = (br['treated'] / br['control']
                        ) * (totalcontrols / totaltreated)
br.loc[(br['leg_black'] == 0), 'weight'] = br['controlweights']

# Now, use the weights to estimate the effect
m = sm.wls(formula='''responded ~ leg_black*treat_out + 
nonblacknonwhite + black_medianhh + white_medianhh + 
statessquireindex + totalpop + urbanpercent''',
           weights=br['weight'],
           data=br).fit()
m.summary()
