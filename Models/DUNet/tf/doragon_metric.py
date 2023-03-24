#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :doragon_metric.py
@Author  :JackHCC
@Date    :2023/3/24 21:40 
@Desc    :

'''
import tensorflow as tf


class Y_AUC(tf.keras.metrics.Metric):

    def __init__(self, num_act, name='y_auc', **kwargs):
        super(Y_AUC, self).__init__(name=name, **kwargs)
        self.auc = tf.keras.metrics.AUC()
        self.num_act = num_act

    def update_state(self, y_true, y_pred, sample_weight=None):
        gain = tf.cast(y_true[:, self.num_act], tf.float32)
        T = y_true[:, :self.num_act]
        gain_pred = tf.squeeze(tf.gather(y_pred[:, :, 1:], indices=tf.math.argmax(T, axis=1), axis=1, batch_dims=1),
                               axis=1)
        gain_pred_proba = tf.keras.activations.sigmoid(gain_pred)
        self.auc.update_state(gain, gain_pred_proba)

    def result(self):
        self.auc.true_positives
        return self.auc.result()

    def reset_states(self):
        self.auc.reset_states()


def y_quantile_metric_proto(y_true, y_pred, num_act, num_quantiles, y_ind):
    gain = tf.cast(y_true[:, y_ind + 1], tf.float32)
    T = y_true[:, :num_act]

    start = 1 + y_ind * num_quantiles
    end = 1 + (y_ind + 1) * num_quantiles
    median = num_quantiles // 2

    gain_pred = tf.gather(y_pred[:, :, start:end], indices=tf.math.argmax(T, axis=1), axis=1, batch_dims=1)

    lossy = tf.keras.metrics.mean_squared_error(gain, gain_pred[:, median])

    return lossy


def y_normal_metric_proto(y_true, y_pred, num_act):
    gain = tf.cast(y_true[:, num_act], tf.float32)
    T = y_true[:, :num_act]

    gain_pred = tf.gather(y_pred[:, :, 1:], indices=tf.math.argmax(T, axis=1), axis=1, batch_dims=1)

    lossy = tf.keras.metrics.mean_squared_error(gain, gain_pred)

    return lossy


def t_metric_proto(y_true, y_pred, num_act):
    gain = tf.cast(y_true[:, num_act], tf.float32)
    T = y_true[:, :num_act]

    T_pred = y_pred[:, :, 0]

    losst = tf.reduce_mean(tf.keras.metrics.categorical_crossentropy(T, T_pred))

    return losst


def respond_loss_proto_(y_true, y_pred, sample_weight, y_loss_func, t_loss_func, num_act, num_quantiles, num_response=1,
                        lamb=1):
    T = tf.one_hot(tf.cast(y_true[:, 0], dtype=tf.int32), num_act)
    Y = tf.expand_dims(tf.cast(y_true[:, 1:], tf.float32), axis=2)

    T_pred = y_pred[:, :, 0]
    Y_pred = tf.gather(y_pred[:, :, 1:],
                       indices=tf.math.argmax(T, axis=1),
                       axis=1, batch_dims=1)
    Y_pred = tf.reshape(Y_pred, [Y_pred.shape[0], num_response, num_quantiles])
    loss_y = y_loss_func(Y, Y_pred)

    loss_t = t_loss_func(T, T_pred)
    return loss_y + lamb * loss_t


def multi_pinball_loss(y_true, y_pred, taus, weights=None):
    error = y_true - y_pred
    if weights is None:
        weights = tf.ones(y_pred.shape[1])
    else:
        weights = tf.constant(weights, dtype=tf.float32)
    loss_single = tf.reduce_sum(tf.maximum(taus * error, (taus - 1) * error), axis=2)
    return tf.reduce_sum(loss_single * weights, axis=1)
