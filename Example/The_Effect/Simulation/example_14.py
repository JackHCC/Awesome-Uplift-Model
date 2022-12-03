#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_14.py
@Author  :JackHCC
@Date    :2022/12/3 17:40 
@Desc    :

'''
import pandas as pd
import numpy as np
from resample.bootstrap import bootstrap
import statsmodels.formula.api as smf
import seaborn as sns

# Get our data to bootstrap
iris = sns.load_dataset('iris')


# Estimation - the first argument should be for the bootstrapped data
def est_model(d):
    # bootstrap() makes an array, not a DataFrame
    d = pd.DataFrame(d)
    # Oh also it tossed out the column names
    d.columns = iris.columns
    # And numeric types
    d = d.convert_dtypes()
    print(d.dtypes)
    m = smf.ols(formula='sepal_length ~ sepal_width + petal_length',
                data=d).fit()
    coefs = [m.params['sepal_width'], m.params['petal_length']]
    return (coefs)


# Bootstrap the iris data, estimate with est_model, and do it 1000 times
b = bootstrap(sample=iris, fn=est_model, size=1000)

# Get our standard errors
bDF = pd.DataFrame(b, columns=['SW', 'PL'])
bDF.std()
