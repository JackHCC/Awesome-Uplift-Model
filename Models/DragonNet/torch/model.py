import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DragonNetBase(nn.Module):
    """
    Base Dragonnet model.

    Parameters
    ----------
    input_dim: int
        input dimension for convariates
    shared_hidden: int
        layer size for hidden shared representation layers
    outcome_hidden: int
        layer size for conditional outcome layers
    """
    def __init__(self, input_dim, shared_hidden=200, outcome_hidden=100):
        super(DragonNetBase, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=shared_hidden)
        self.fc2 = nn.Linear(in_features=shared_hidden, out_features=shared_hidden)
        self.fcz = nn.Linear(in_features=shared_hidden, out_features=shared_hidden)

        self.treat_out = nn.Linear(in_features=shared_hidden, out_features=1)

        self.y0_fc1 = nn.Linear(in_features=shared_hidden, out_features=outcome_hidden)
        self.y0_fc2 = nn.Linear(in_features=outcome_hidden, out_features=outcome_hidden)
        self.y0_out = nn.Linear(in_features=outcome_hidden, out_features=1)

        self.y1_fc1 = nn.Linear(in_features=shared_hidden, out_features=outcome_hidden)
        self.y1_fc2 = nn.Linear(in_features=outcome_hidden, out_features=outcome_hidden)
        self.y1_out = nn.Linear(in_features=outcome_hidden, out_features=1)

        self.epsilon = nn.Linear(in_features=1, out_features=1)
        torch.nn.init.xavier_normal_(self.epsilon.weight)

    def forward(self, inputs):
        """
        forward method to train model.

        Parameters
        ----------
        inputs: torch.Tensor
            covariates

        Returns
        -------
        y0: torch.Tensor
            outcome under control
        y1: torch.Tensor
            outcome under treatment
        t_pred: torch.Tensor
            predicted treatment
        eps: torch.Tensor
            trainable epsilon parameter
        """
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        z = F.relu(self.fcz(x))

        t_pred = torch.sigmoid(self.treat_out(z))

        y0 = F.relu(self.y0_fc1(z))
        y0 = F.relu(self.y0_fc2(y0))
        y0 = self.y0_out(y0)

        y1 = F.relu(self.y1_fc1(z))
        y1 = F.relu(self.y1_fc2(y1))
        y1 = self.y1_out(y1)

        eps = self.epsilon(torch.ones_like(t_pred)[:, 0:1])

        return y0, y1, t_pred, eps


def dragonnet_loss(y_true, t_true, t_pred, y0_pred, y1_pred, eps, alpha=1.0):
    """
    Generic loss function for dragonnet

    Parameters
    ----------
    y_true: torch.Tensor
        Actual target variable
    t_true: torch.Tensor
        Actual treatment variable
    t_pred: torch.Tensor
        Predicted treatment
    y0_pred: torch.Tensor
        Predicted target variable under control
    y1_pred: torch.Tensor
        Predicted target variable under treatment
    eps: torch.Tensor
        Trainable epsilon parameter
    alpha: float
        loss component weighting hyperparameter between 0 and 1
    Returns
    -------
    loss: torch.Tensor
    """
    t_pred = (t_pred + 0.01) / 1.02
    loss_t = torch.sum(F.binary_cross_entropy(t_pred, t_true))

    loss0 = torch.sum((1. - t_true) * torch.square(y_true - y0_pred))
    loss1 = torch.sum(t_true * torch.square(y_true - y1_pred))
    loss_y = loss0 + loss1

    loss = loss_y + alpha * loss_t

    return loss


def tarreg_loss(y_true, t_true, t_pred, y0_pred, y1_pred, eps, alpha=1.0, beta=1.0):
    """
    Targeted regularisation loss function for dragonnet

    Parameters
    ----------
    y_true: torch.Tensor
        Actual target variable
    t_true: torch.Tensor
        Actual treatment variable
    t_pred: torch.Tensor
        Predicted treatment
    y0_pred: torch.Tensor
        Predicted target variable under control
    y1_pred: torch.Tensor
        Predicted target variable under treatment
    eps: torch.Tensor
        Trainable epsilon parameter
    alpha: float
        loss component weighting hyperparameter between 0 and 1
    beta: float
        targeted regularization hyperparameter between 0 and 1
    Returns
    -------
    loss: torch.Tensor
    """
    vanilla_loss = dragonnet_loss(y_true, t_true, t_pred, y0_pred, y1_pred, alpha)
    t_pred = (t_pred + 0.01) / 1.02

    y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

    h = (t_true / t_pred) - ((1 - t_true) / (1 - t_pred))

    y_pert = y_pred + eps * h
    targeted_regularization = torch.sum((y_true - y_pert)**2)

    # final
    loss = vanilla_loss + beta * targeted_regularization
    return loss


class EarlyStopper:
    def __init__(self, patience=15, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

