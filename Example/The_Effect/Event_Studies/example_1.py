#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_1.py
@Author  :JackHCC
@Date    :2022/12/3 17:46 
@Desc    :

'''
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
from causaldata import google_stock

goog = google_stock.load_pandas().data

# Create estimation data set
goog['Date'] = pd.to_datetime(goog['Date'])
est_data = goog.loc[(goog['Date'] >= pd.Timestamp(2015, 5, 1)) &
                    (goog['Date'] < pd.Timestamp(2015, 7, 31))]

# And observation data
obs_data = goog.loc[(goog['Date'] >= pd.Timestamp(2015, 8, 6)) &
                    (goog['Date'] < pd.Timestamp(2015, 8, 24))]

# Estimate a model predicting stock price with market return
m = smf.ols('Google_Return ~ SP500_Return', data=est_data).fit()

# Get AR
# Using mean of estimation return
goog_return = np.mean(est_data['Google_Return'])
obs_data['AR_mean'] = obs_data['Google_Return'] - goog_return
# Then comparing to market return
obs_data['AR_market'] = obs_data['Google_Return'] - obs_data['SP500_Return']
# Then using model fit with estimation data
obs_data['risk_pred'] = m.predict(obs_data)
obs_data['AR_risk'] = obs_data['Google_Return'] - obs_data['risk_pred']

# Graph the results
sns.lineplot(x=obs_data['Date'], y=obs_data['AR_risk'])
plt.axvline(pd.Timestamp(2015, 8, 10))
plt.axhline(0)
plt.show()
