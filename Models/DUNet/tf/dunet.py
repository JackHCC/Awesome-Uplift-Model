#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :dunet.py
@Author  :JackHCC
@Date    :2023/3/24 20:41 
@Desc    :

'''

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers.experimental.preprocessing import Hashing
from tensorflow.keras.layers import BatchNormalization, Dense, Input, Lambda


class QuantileActTower(layers.Layer):
    def __init__(self, tower_config=[256, 128, 64], monotonic=False, reverse=False, num_quantiles=20):
        """
        ActTower for y-Quantiles

        Return Shape:
            value head:    [Batch, 1]
            quantile head: [Batch, NumQuantiles]

        Args:
            tower_config (list, optional): [Dense config]. Defaults to [256, 128, 64].
            monotonic (bool, optional): [If True, please ensure the correlation between y and t]. Defaults to False.
            num_quantiles (int, optional): [Numbers of quantiles]. Defaults to 20.
        """
        super(QuantileActTower, self).__init__()
        initializer = tf.keras.initializers.HeUniform()

        self.num_quantiles = num_quantiles
        self.median_ind = num_quantiles // 2
        self.tower_config = tower_config
        self.monotonic = monotonic

        self.dense_layers = tf.keras.Sequential()
        for out_dim in tower_config:
            self.dense_layers.add(
                Dense(out_dim, activation='relu', kernel_initializer=initializer))

        if self.monotonic:
            self.value_head = Dense(
                1, activation='elu', kernel_initializer=initializer)
        else:
            self.value_head = Dense(1, kernel_initializer=initializer)

        self.quantile_head = Dense(
            num_quantiles, activation='elu', kernel_initializer=initializer)

    def call(self, inputs):
        dense_out = self.dense_layers(inputs)

        value_out = self.value_head(dense_out)

        quantile_out = self.quantile_head(dense_out) + 1
        quantile_out = tf.cumsum(quantile_out, axis=1)
        quantile_out = quantile_out - \
                       quantile_out[:, self.median_ind:self.median_ind + 1]
        if self.monotonic:
            value_out = value_out + 1

        return value_out, quantile_out


class QuantileDragonNet(Model):
    def __init__(self, num_action=5, monotonic=False, reverse=False, num_quantiles=20, act_tower_config=[64, 64]):
        """
        Qiantile DragonNet.

        Return Shape:[Batch, [NumActions, NumQiantiles]]

        Args:
            num_action (int, optional): [description]. Defaults to 6.
            monotonic (bool, optional): [description]. Defaults to False.
            num_quantiles (int, optional): [description]. Defaults to 20.
            base_net ([type], optional): [description]. Defaults to Dense(20).
            act_tower_config (list, optional): [description]. Defaults to [64, 64].
        """
        super(QuantileDragonNet, self).__init__()
        self.num_quantiles = num_quantiles
        self.monotonic = monotonic
        self.reverse = reverse

        self.act_towers = [
            QuantileActTower(act_tower_config, monotonic=False, num_quantiles=num_quantiles)]

        for _ in range(num_action - 1):
            self.act_towers.append(QuantileActTower(
                act_tower_config, monotonic=monotonic, reverse=reverse, num_quantiles=num_quantiles))

    def call(self, inputs):
        action_outs = [tower(inputs) for tower in self.act_towers]
        delta_values = tf.stack([x[0] for x in action_outs], axis=1)
        delta_quantiles = tf.stack([x[1] for x in action_outs], axis=1)

        if self.monotonic:
            values = tf.math.cumsum(delta_values, axis=1, reverse=self.reverse)
        else:
            values = delta_values

        output = values + delta_quantiles
        return output


class NormalActTower(layers.Layer):
    def __init__(self, tower_config=[256, 128, 64], monotonic=False, reverse=False):
        """
        Tower for y

        Args:
            tower_config (list, optional): [Dense config]. Defaults to [256, 128, 64].
            monotonic (bool, optional): [If True, please ensure the correlation between y and t]. Defaults to False.
        """
        super(NormalActTower, self).__init__()
        initializer = tf.keras.initializers.HeUniform()
        self.tower_config = tower_config
        self.monotonic = monotonic
        self.dense_layers = tf.keras.Sequential()
        for out_dim in tower_config:
            self.dense_layers.add(
                Dense(out_dim, activation='relu', kernel_initializer=initializer))
        if self.monotonic:
            self.value_head = Dense(
                1, activation='elu', kernel_initializer=initializer)
        else:
            self.value_head = Dense(1, kernel_initializer=initializer)

    def call(self, inputs):
        dense_out = self.dense_layers(inputs)
        value_out = self.value_head(dense_out)
        if self.monotonic:
            value_out = value_out + 1

        return value_out


class NormalDragonNet(layers.Layer):
    def __init__(self, num_action=6, monotonic=False, reverse=False, act_tower_config=[64, 64]):
        """
        Nomal regression y without quantile. Use normal MSE or MAE loss function to y.

        Args:
            num_action (int, optional): [Numbers of treatments]. Defaults to 6.
            monotonic (bool, optional): [If True, please ensure the correlation between y and t]. Defaults to False.
            base_net ([type], optional): [Define your own base net]. Defaults to Dense(20).
            act_tower_config (list, optional): [Dense config]. Defaults to [64, 64].
        """
        super(NormalDragonNet, self).__init__()
        self.monotonic = monotonic
        self.reverse = reverse

        self.act_towers = []
        for _ in range(num_action):
            self.act_towers.append(NormalActTower(
                tower_config=act_tower_config, monotonic=monotonic, reverse=reverse))

    def call(self, inputs):
        action_outs = [tower(inputs) for tower in self.act_towers]
        if self.monotonic:
            values = tf.math.cumsum(action_outs, reverse=self.reverse)
        else:
            values = action_outs
        output = tf.keras.backend.permute_dimensions(values, (1, 0, 2))

        return output


class DragonNet(layers.Layer):
    def __init__(self, num_action=6, monotonic=False, reverse=False, num_quantiles=20, base_net=Dense(20),
                 act_tower_config=[64, 64], g_layer=None):
        """
        Doragon Neto

        Args:
            num_action (int, optional): [Numbers of treatment]. Defaults to 6.
            monotonic (bool, optional): [If True, please ensure the correlation between y and t]. Defaults to False.
            num_quantiles (int, optional): [Use y-Quantiles instead of y. Please set 1 if you don't need y-Quantiles.]. Defaults to 20.
            base_net ([type], optional): [Define your own base net]. Defaults to Dense(20).
            act_tower_config (list, optional): [Dense config]. Defaults to [64, 64].
            g_layer (layers.Layer,optional): [description]. Defaults to None.
        """
        super(DragonNet, self).__init__()
        if num_quantiles <= 1:
            self.model = NormalDragonNet(
                num_action=num_action, monotonic=monotonic, reverse=reverse, act_tower_config=act_tower_config)
        else:
            self.model = QuantileDragonNet(
                num_action=num_action, monotonic=monotonic, reverse=reverse, num_quantiles=num_quantiles,
                act_tower_config=act_tower_config)

    def call(self, inputs):
        return self.model(inputs)
