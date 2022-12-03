#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_3.py
@Author  :JackHCC
@Date    :2022/12/3 17:51 
@Desc    :

'''
import pandas as pd
import matplotlib.pyplot as plt
import linearmodels as lm
from causaldata import organ_donations

od = organ_donations.load_pandas().data

# Create Treatment Variable
od['California'] = od['State'] == 'California'

# Create our interactions by hand,
# skipping quarter 3, the last one before treatment
for i in [1, 2, 4, 5, 6]:
    name = 'INX' + str(i)
    od[name] = 1 * od['California']
    od.loc[od['Quarter_Num'] != i, name] = 0

# Set our individual and time (index) for our data
od = od.set_index(['State', 'Quarter_Num'])

mod = lm.PanelOLS.from_formula('''Rate ~ 
INX1 + INX2 + INX4 + INX5 + INX6 + 
EntityEffects + TimeEffects''', od)

# Specify clustering when we fit the model
clfe = mod.fit(cov_type='clustered',
               cluster_entity=True)

# Get coefficients and CIs
res = pd.concat([clfe.params, clfe.std_errors], axis=1)
# Scale standard error to CI
res['ci'] = res['std_error'] * 1.96

# Add our quarter values
res['Quarter_Num'] = [1, 2, 4, 5, 6]
# And add our reference period back in
reference = pd.DataFrame([[0, 0, 0, 3]],
                         columns=['parameter',
                                  'lower',
                                  'upper',
                                  'Quarter_Num'])
res = pd.concat([res, reference])

# For plotting, sort and add labels
res = res.sort_values('Quarter_Num')
res['Quarter'] = ['Q42010', 'Q12011',
                  'Q22011', 'Q32011',
                  'Q42011', 'Q12012']

# Plot the estimates as connected lines with error bars

plt.errorbar(x='Quarter', y='parameter',
                    yerr='ci', data=res)
# Add a horizontal line at 0
plt.axhline(0, linestyle='dashed')
plt.show()
