#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_2.py
@Author  :JackHCC
@Date    :2022/12/3 17:49 
@Desc    :

'''
import linearmodels as lm
from causaldata import organ_donations

od = organ_donations.load_pandas().data

# Keep only pre-treatment data
od = od.loc[od['Quarter_Num'] <= 3]

# Create fake treatment variables
od['California'] = od['State'] == 'California'
od['FakeAfter1'] = od['Quarter_Num'] > 1
od['FakeAfter2'] = od['Quarter_Num'] > 2
od['FakeTreat1'] = 1 * (od['California'] & od['FakeAfter1'])
od['FakeTreat2'] = 1 * (od['California'] & od['FakeAfter2'])

# Set our individual and time (index) for our data
od = od.set_index(['State', 'Quarter_Num'])

# Run the same model as before
# but with our fake treatment variables
mod1 = lm.PanelOLS.from_formula('''Rate ~ 
FakeTreat1 + EntityEffects + TimeEffects''', od)
mod2 = lm.PanelOLS.from_formula('''Rate ~ 
FakeTreat2 + EntityEffects + TimeEffects''', od)

clfe1 = mod1.fit(cov_type='clustered',
                 cluster_entity=True)
clfe2 = mod1.fit(cov_type='clustered',
                 cluster_entity=True)

print(clfe1)
print(clfe2)
