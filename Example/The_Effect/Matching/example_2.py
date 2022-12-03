#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_2.py
@Author  :JackHCC
@Date    :2022/12/3 17:26 
@Desc    :

'''
import pandas as pd
import statsmodels.formula.api as sm
# There is a cem package but it doesn't seem to work that well
# So we will do this by hand
from causaldata import black_politicians

br = black_politicians.load_pandas().data

# Create bins for our continuous matching variables
# cut creates evenly spaced bins
# while qcut cuts based on quantiles
br['inc_bins'] = pd.qcut(br['medianhhincom'], 6)
br['bp_bins'] = pd.qcut(br['blackpercent'], 6)

# Count how many treated and control observations
# are in each bin
treated = br.loc[br['leg_black'] == 1
                 ].groupby(['inc_bins', 'bp_bins', 'leg_democrat']
                           ).size().to_frame('treated')
control = br.loc[br['leg_black'] == 0
                 ].groupby(['inc_bins', 'bp_bins', 'leg_democrat']
                           ).size().to_frame('control')

# Merge those counts back in
br = br.join(treated, on=['inc_bins', 'bp_bins', 'leg_democrat'])
br = br.join(control, on=['inc_bins', 'bp_bins', 'leg_democrat'])

# For treated obs, weight is 1 if there are any control matches
br['weight'] = 0
br.loc[(br['leg_black'] == 1) & (br['control'] > 0), 'weight'] = 1
# For control obs, weight depends on total number of treated and control
# obs that found matches
totalcontrols = sum(br.loc[br['leg_black'] == 0]['treated'] > 0)
totaltreated = sum(br.loc[br['leg_black'] == 1]['control'] > 0)
# Then, control weights are treated/control in the bin,
# times control/treated overall
br['controlweights'] = (br['treated'] / br['control']
                        ) * (totalcontrols / totaltreated)
br.loc[(br['leg_black'] == 0), 'weight'] = br['controlweights']

# Now, use the weights to estimate the effect
m = sm.wls(formula='responded ~ leg_black',
           weights=br['controlweights'],
           data=br).fit()

m.summary()
