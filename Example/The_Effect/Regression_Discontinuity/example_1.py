#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example_1.py
@Author  :JackHCC
@Date    :2022/12/3 17:54 
@Desc    :

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from causaldata import gov_transfers

d = gov_transfers.load_pandas().data

# cut at 0, and 15 places on either side
edges = np.linspace(-.02, .02, 31)
d['Bins'] = pd.cut(d['Income_Centered'], bins=edges)

# Mean within bins
binned = d.groupby(['Bins']).agg('mean')

# And plot
sns.lineplot(x=binned['Income_Centered'],
             y=binned['Support'])
# Add vertical line at cutoff
plt.axvline(0, 0, 1)
plt.show()
