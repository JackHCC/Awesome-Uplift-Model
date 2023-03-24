#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :initialisation.py
@Author  :JackHCC
@Date    :2023/3/24 21:53 
@Desc    :

'''
import numpy as np
from torch import optim
import torch
import torch.distributions


def init_qz(qz, pz, y, t, x):
    """
    Initialize qz towards outputting standard normal distributions
    - with standard torch init of weights the gradients tend to explode after first update step
    """
    idx = list(range(x.shape[0]))
    np.random.shuffle(idx)

    optimizer = optim.Adam(qz.parameters(), lr=0.001)

    for i in range(50):
        batch = np.random.choice(idx, 1)
        x_train, y_train, t_train = torch.cuda.FloatTensor(x[batch]), torch.cuda.FloatTensor(y[batch]), \
            torch.cuda.FloatTensor(t[batch])
        xy = torch.cat((x_train, y_train), 1)

        z_infer = qz(xy=xy, t=t_train)

        # KL(q_z|p_z) mean approx, to be minimized
        # KLqp = (z_infer.log_prob(z_infer.mean) - pz.log_prob(z_infer.mean)).sum(1)
        # Analytic KL
        KLqp = (-torch.log(z_infer.stddev) + 1 / 2 * (z_infer.variance + z_infer.mean ** 2 - 1)).sum(1)

        objective = KLqp
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

        if KLqp != KLqp:
            raise ValueError('KL(pz,qz) contains NaN during init')

    return qz
