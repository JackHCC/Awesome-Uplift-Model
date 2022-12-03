#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_1.py
@Author  :JackHCC
@Date    :2022/12/3 17:48 
@Desc    :

'''
import linearmodels as lm
from causaldata import organ_donations

od = organ_donations.load_pandas().data

# Create Treatment Variable
od['California'] = od['State'] == 'California'
od['After'] = od['Quarter_Num'] > 3
od['Treated'] = 1 * (od['California'] & od['After'])
# Set our individual and time (index) for our data
od = od.set_index(['State', 'Quarter_Num'])

mod = lm.PanelOLS.from_formula('''Rate ~ 
Treated + EntityEffects + TimeEffects''', od)

# Specify clustering when we fit the model
clfe = mod.fit(cov_type='clustered',
               cluster_entity=True)
print(clfe)
