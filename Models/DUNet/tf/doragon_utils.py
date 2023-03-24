#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :doragon_utils.py
@Author  :JackHCC
@Date    :2023/3/24 21:40 
@Desc    :

'''
import numpy as np
import pandas as pd
from sklearn.utils.extmath import stable_cumsum
from functools import partial, update_wrapper


def AUUCScorer(y_true, uplift, treatment):
    desc_score_indices = np.argsort(uplift, kind="mergesort")[::-1]
    y_true, uplift, treatment = y_true[desc_score_indices], uplift[desc_score_indices], treatment[desc_score_indices]
    y_true_ctrl, y_true_trmnt = y_true.copy(), y_true.copy()

    y_true_ctrl[treatment == 1] = 0
    y_true_trmnt[treatment == 0] = 0

    ctrl = 1 - treatment

    distinct_value_indices = np.arange(len(uplift))
    threshold_indices = np.r_[distinct_value_indices, uplift.size - 1]
    num_all = threshold_indices + 1

    num_trmnt = stable_cumsum(treatment)[threshold_indices]
    y_trmnt = stable_cumsum(y_true_trmnt)[threshold_indices]

    num_ctrl = stable_cumsum(ctrl)[threshold_indices]

    y_ctrl = stable_cumsum(y_true_ctrl)[threshold_indices]

    avg_trmnt = np.divide(y_trmnt, num_trmnt, out=np.zeros_like(y_trmnt), where=num_trmnt != 0)

    avg_ctrl = np.divide(y_ctrl, num_ctrl, out=np.zeros_like(y_ctrl), where=num_ctrl != 0)

    curve_values = (avg_trmnt - avg_ctrl) * (threshold_indices + 1)
    lift = pd.DataFrame()
    lift['ATE'] = curve_values
    lift['random'] = np.linspace(0, lift['ATE'][len(curve_values) - 1], len(curve_values))

    auuc = {}
    auuc['ATE'] = (lift['ATE'].sum() - lift['ATE'].iloc[-1] / 2) / lift.shape[0]
    auuc['random'] = (lift['random'].sum() - lift['random'].iloc[-1] / 2) / lift.shape[0]

    return lift, auuc


def wrapped_partial(func, name, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    partial_func.__name__ = name
    return partial_func
