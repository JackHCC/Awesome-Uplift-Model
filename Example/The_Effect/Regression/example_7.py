#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_7.py
@Author  :JackHCC
@Date    :2022/12/3 17:21 
@Desc    :

'''
import statsmodels.formula.api as sm
from causaldata import restaurant_inspections

df = restaurant_inspections.load_pandas().data

# Aggregate the data
df['Num_Inspections'] = 1
df = df.groupby('business_name').agg(
    {'inspection_score': 'mean',
     'Year': 'min',
     'Num_Inspections': 'sum'})

# Here we call a special WLS function
m1 = sm.wls(formula='inspection_score ~ Year',
            weights=df['Num_Inspections'],
            data=df).fit()

m1.summary()
