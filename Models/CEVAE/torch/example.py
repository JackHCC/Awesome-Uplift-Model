#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Awesome-Uplift-Model 
@File    :example.py
@Author  :JackHCC
@Date    :2023/3/24 21:53 
@Desc    :

'''
from argparse import ArgumentParser

from initialisation import init_qz
from datasets import IHDP
from evaluation import Evaluator, get_y0_y1
from Models.CEVAE.torch.cevae import p_x_z, p_t_z, p_y_zt, q_t_x, q_y_xt, q_z_tyx

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
from torch.distributions import normal
from torch import optim

# set random seeds:
# torch.manual_seed(7)
# np.random.seed(7)

parser = ArgumentParser()

# Set Hyperparameters
parser.add_argument('-reps', type=int, default=1)
parser.add_argument('-z_dim', type=int, default=20)
parser.add_argument('-h_dim', type=int, default=64)
parser.add_argument('-epochs', type=int, default=100)
parser.add_argument('-batch', type=int, default=100)
parser.add_argument('-lr', type=float, default=0.00001)
parser.add_argument('-decay', type=float, default=0.001)
parser.add_argument('-print_every', type=int, default=10)

args = parser.parse_args()

dataset = IHDP(replications=args.reps)

# Loop for replications
for i, (train, valid, test, contfeats, binfeats) in enumerate(dataset.get_train_valid_test()):
    print('\nReplication %i/%i' % (i + 1, args.reps))
    # read out data
    (xtr, ttr, ytr), (y_cftr, mu0tr, mu1tr) = train
    (xva, tva, yva), (y_cfva, mu0va, mu1va) = valid
    (xte, tte, yte), (y_cfte, mu0te, mu1te) = test

    # reorder features with binary first and continuous after
    perm = binfeats + contfeats
    xtr, xva, xte = xtr[:, perm], xva[:, perm], xte[:, perm]
    # concatenate train and valid for training
    xalltr, talltr, yalltr = np.concatenate([xtr, xva], axis=0), np.concatenate([ttr, tva], axis=0), np.concatenate(
        [ytr, yva], axis=0)

    # set evaluator objects
    evaluator_train = Evaluator(yalltr, talltr, y_cf=np.concatenate([y_cftr, y_cfva], axis=0),
                                mu0=np.concatenate([mu0tr, mu0va], axis=0), mu1=np.concatenate([mu1tr, mu1va], axis=0))
    evaluator_test = Evaluator(yte, tte, y_cf=y_cfte, mu0=mu0te, mu1=mu1te)

    # zero mean, unit variance for y during training, use ym & ys to correct when using testset
    ym, ys = np.mean(ytr), np.std(ytr)
    ytr, yva = (ytr - ym) / ys, (yva - ym) / ys

    # init networks (overwritten per replication)
    x_dim = len(binfeats) + len(contfeats)
    p_x_z_dist = p_x_z(dim_in=args.z_dim, nh=3, dim_h=args.h_dim, dim_out_bin=len(binfeats),
                       dim_out_con=len(contfeats)).cuda()
    p_t_z_dist = p_t_z(dim_in=args.z_dim, nh=1, dim_h=args.h_dim, dim_out=1).cuda()
    p_y_zt_dist = p_y_zt(dim_in=args.z_dim, nh=3, dim_h=args.h_dim, dim_out=1).cuda()
    q_t_x_dist = q_t_x(dim_in=x_dim, nh=1, dim_h=args.h_dim, dim_out=1).cuda()
    # t is not feed into network, therefore not increasing input size (y is fed).
    q_y_xt_dist = q_y_xt(dim_in=x_dim, nh=3, dim_h=args.h_dim, dim_out=1).cuda()
    q_z_tyx_dist = q_z_tyx(dim_in=len(binfeats) + len(contfeats) + 1, nh=3, dim_h=args.h_dim,
                           dim_out=args.z_dim).cuda()
    p_z_dist = normal.Normal(torch.zeros(args.z_dim).cuda(), torch.ones(args.z_dim).cuda())

    # Create optimizer
    params = list(p_x_z_dist.parameters()) + \
             list(p_t_z_dist.parameters()) + \
             list(p_y_zt_dist.parameters()) + \
             list(q_t_x_dist.parameters()) + \
             list(q_y_xt_dist.parameters()) + \
             list(q_z_tyx_dist.parameters())

    # Adam is used, like original implementation, in paper Adamax is suggested
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.decay)

    # init q_z inference
    q_z_tyx_dist = init_qz(q_z_tyx_dist, p_z_dist, ytr, ttr, xtr)

    # set batch size
    M = args.batch

    n_epoch, n_iter_per_epoch, idx = args.epochs, 10 * int(xtr.shape[0] / M), list(range(xtr.shape[0]))

    loss = defaultdict(list)
    for epoch in range(n_epoch):
        # print('Epoch: %i/%i' % (epoch, n_epoch))
        loss_sum = 0.
        # shuffle index
        np.random.shuffle(idx)
        # take random batch for training
        for j in range(n_iter_per_epoch):
            # select random batch
            batch = np.random.choice(idx, M)
            x_train, y_train, t_train = torch.cuda.FloatTensor(xalltr[batch]), torch.cuda.FloatTensor(yalltr[batch]), \
                torch.cuda.FloatTensor(talltr[batch])

            # inferred distribution over z
            xy = torch.cat((x_train, y_train), 1)
            z_infer = q_z_tyx_dist(xy=xy, t=t_train)
            # use a single sample to approximate expectation in lowerbound
            z_infer_sample = z_infer.sample()

            # RECONSTRUCTION LOSS
            # p(x|z)
            x_bin, x_con = p_x_z_dist(z_infer_sample)
            l1 = x_bin.log_prob(x_train[:, :len(binfeats)]).sum(1)
            loss['Reconstr_x_bin'].append(l1.sum().cpu().detach().float())
            l2 = x_con.log_prob(x_train[:, -len(contfeats):]).sum(1)
            loss['Reconstr_x_con'].append(l2.sum().cpu().detach().float())
            # p(t|z)
            t = p_t_z_dist(z_infer_sample)
            l3 = t.log_prob(t_train).squeeze()
            loss['Reconstr_t'].append(l3.sum().cpu().detach().float())
            # p(y|t,z)
            # for training use t_train, in out-of-sample prediction this becomes t_infer
            y = p_y_zt_dist(z_infer_sample, t_train)
            l4 = y.log_prob(y_train).squeeze()
            loss['Reconstr_y'].append(l4.sum().cpu().detach().float())

            # REGULARIZATION LOSS
            # p(z) - q(z|x,t,y)
            # approximate KL
            l5 = (p_z_dist.log_prob(z_infer_sample) - z_infer.log_prob(z_infer_sample)).sum(1)
            # Analytic KL (seems to make overall performance less stable)
            # l5 = (-torch.log(z_infer.stddev) + 1/2*(z_infer.variance + z_infer.mean**2 - 1)).sum(1)
            loss['Regularization'].append(l5.sum().cpu().detach().float())

            # AUXILIARY LOSS
            # q(t|x)
            t_infer = q_t_x_dist(x_train)
            l6 = t_infer.log_prob(t_train).squeeze()
            loss['Auxiliary_t'].append(l6.sum().cpu().detach().float())
            # q(y|x,t)
            y_infer = q_y_xt_dist(x_train, t_train)
            l7 = y_infer.log_prob(y_train).squeeze()
            loss['Auxiliary_y'].append(l7.sum().cpu().detach().float())

            # Total objective
            # inner sum to calculate loss per item, torch.mean over batch
            loss_mean = torch.mean(l1 + l2 + l3 + l4 + l5 + l6 + l7)
            loss['Total'].append(loss_mean.cpu().detach().numpy())
            objective = -loss_mean

            optimizer.zero_grad()
            # Calculate gradients
            objective.backward()
            # Update step
            optimizer.step()

        if epoch % args.print_every == 0:
            print('Epoch %i' % epoch)
            y0, y1 = get_y0_y1(p_y_zt_dist, q_y_xt_dist, q_z_tyx_dist, torch.tensor(xalltr).cuda(),
                               torch.tensor(talltr).cuda())
            score_train = evaluator_train.calc_stats(y1, y0)
            rmses_train = evaluator_train.y_errors(y0, y1)
            print('Training set - ite: %f, ate: %f, pehe: %f' % score_train)
            print('Training set - rmse factual: %f, rmse counterfactual: %f' % rmses_train)

            y0, y1 = get_y0_y1(p_y_zt_dist, q_y_xt_dist, q_z_tyx_dist, torch.tensor(xte).cuda(),
                               torch.tensor(tte).cuda())
            y0, y1 = y0 * ys + ym, y1 * ys + ym
            score_test = evaluator_test.calc_stats(y1, y0)
            print('Testset - ite: %f, ate: %f, pehe: %f' % score_test)

    plt.figure()
    plt.plot(np.array(loss['Total']), label='Total')
    plt.title('Variational Lower Bound')
    plt.show()

    plt.figure()
    subidx = 1
    for key, value in loss.items():
        plt.subplot(2, 4, subidx)
        plt.plot(np.array(value), label=key)
        plt.title(key)
        subidx += 1
    plt.show()
