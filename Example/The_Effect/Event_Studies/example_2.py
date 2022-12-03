#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_2.py
@Author  :JackHCC
@Date    :2022/12/3 17:47 
@Desc    :

'''
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

np.random.seed(10)

# Ten groups with ten periods each
id = pd.DataFrame({'id': range(0, 10), 'key': 1})
t = pd.DataFrame({'t': range(1, 11), 'key': 1})
d = id.merge(t, on='key')
# Add an event in period 6 with a one-period effect
d['Y'] = np.random.normal(0, 1, 100) + 1 * (d['t'] == 6)

# Estimate our model using time 5 as reference
m = smf.ols('Y~C(t, Treatment(reference = 5))', data=d)

# Fit with SEs clustered at the group level
m = m.fit(cov_type='cluster', cov_kwds={'groups': d['id']})

# Get coefficients and CIs
# The original table will have an intercept up top
# But we'll overwrite it with our 5 reference
p = pd.DataFrame({'t': [5, 1, 2, 3, 4, 6, 7, 8, 9, 10],
                  'b': m.params, 'se': m.bse})
# And add our period-5 zero
p.iloc[0] = [5, 0, 0]

# Sort for plotting
p = p.sort_values('t')
# and make CIs by scaling the standard error
p['ci'] = 1.96 * p['se']

# Plot the estimates as connected lines with error bars
plt.errorbar(x='t', y='b',
             yerr='ci', data=p)
# Add a horizontal line at 0
plt.axhline(0, linestyle='dashed')
# And a vertical line at the treatment time
plt.axvline(5)
plt.show()
