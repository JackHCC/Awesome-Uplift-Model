from functools import partial

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from Models.DragonNet.model import DragonNetBase, dragonnet_loss, tarreg_loss, EarlyStopper


class DragonNet:
    """
    Main class for the Dragonnet model

    Parameters
    ----------
    input_dim: int
        input dimension for convariates
    shared_hidden: int, default=200
        layer size for hidden shared representation layers
    outcome_hidden: int, default=100
        layer size for conditional outcome layers
    alpha: float, default=1.0
        loss component weighting hyperparameter between 0 and 1
    beta: float, default=1.0
        targeted regularization hyperparameter between 0 and 1
    epochs: int, default=200
        Number training epochs
    batch_size: int, default=64
        Training batch size
    learning_rate: float, default=1e-3
        Learning rate
    data_loader_num_workers: int, default=4
        Number of workers for data loader
    loss_type: str, {'tarreg', 'default'}, default='tarreg'
        Loss function to use
    """

    def __init__(
        self,
        input_dim,
        shared_hidden=200,
        outcome_hidden=100,
        alpha=1.0,
        beta=1.0,
        epochs=200,
        batch_size=64,
        learning_rate=1e-5,
        data_loader_num_workers=4,
        loss_type="tarreg",
    ):

        self.model = DragonNetBase(input_dim, shared_hidden, outcome_hidden)
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = data_loader_num_workers
        self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.train_dataloader = None
        self.valid_dataloader = None
        if loss_type == "tarreg":
            self.loss_f = partial(tarreg_loss, alpha=alpha, beta=beta)
        elif loss_type == "default":
            self.loss_f = partial(dragonnet_loss, alpha=alpha)

    def create_dataloaders(self, x, y, t, valid_perc=None):
        """
        Utility function to create train and validation data loader:

        Parameters
        ----------
        x: np.array
            covariates
        y: np.array
            target variable
        t: np.array
            treatment
        """
        if valid_perc:
            x_train, x_test, y_train, y_test, t_train, t_test = train_test_split(
                x, y, t, test_size=valid_perc, random_state=42
            )
            x_train = torch.Tensor(x_train)
            x_test = torch.Tensor(x_test)
            y_train = torch.Tensor(y_train).reshape(-1, 1)
            y_test = torch.Tensor(y_test).reshape(-1, 1)
            t_train = torch.Tensor(t_train).reshape(-1, 1)
            t_test = torch.Tensor(t_test).reshape(-1, 1)
            train_dataset = TensorDataset(x_train, t_train, y_train)
            valid_dataset = TensorDataset(x_test, t_test, y_test)
            self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        else:
            x = torch.Tensor(x)
            t = torch.Tensor(t).reshape(-1, 1)
            y = torch.Tensor(y).reshape(-1, 1)
            train_dataset = TensorDataset(x, t, y)
            self.train_dataloader = DataLoader(
                train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
            )

    def fit(self, x, y, t, valid_perc=None):
        """
        Function used to train the dragonnet model

        Parameters
        ----------
        x: np.array
            covariates
        y: np.array
            target variable
        t: np.array
            treatment
        valid_perc: float
            Percentage of data to allocate to validation set
        """
        self.create_dataloaders(x, y, t, valid_perc)
        early_stopper = EarlyStopper(patience=10, min_delta=0)
        for epoch in range(self.epochs):
            for batch, (X, tr, y1) in enumerate(self.train_dataloader):
                y0_pred, y1_pred, t_pred, eps = self.model(X)
                loss = self.loss_f(y1, tr, t_pred, y0_pred, y1_pred, eps)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            if self.valid_dataloader:
                self.model.eval()
                valid_loss = self.validate_step()
                print(
                    f"epoch: {epoch}--------- train_loss: {loss} ----- valid_loss: {valid_loss}"
                )
                self.model.train()
                if early_stopper.early_stop(valid_loss):
                    break
            else:
                print(f"epoch: {epoch}--------- train_loss: {loss}")

    def validate_step(self):
        """
        Calculates validation loss

        Returns
        -------
        valid_loss: torch.Tensor
            validation loss
        """
        valid_loss = []
        with torch.no_grad():
            for batch, (X, tr, y1) in enumerate(self.valid_dataloader):
                y0_pred, y1_pred, t_pred, eps = self.predict(X)
                loss = self.loss_f(y1, tr, t_pred, y0_pred, y1_pred, eps)
                valid_loss.append(loss)
        return torch.Tensor(valid_loss).mean()

    def predict(self, x):
        """
        Function used to predict on covariates.

        Parameters
        ----------
        x: torch.Tensor or numpy.array
            covariates

        Returns
        -------
        y0_pred: torch.Tensor
            outcome under control
        y1_pred: torch.Tensor
            outcome under treatment
        t_pred: torch.Tensor
            predicted treatment
        eps: torch.Tensor
            trainable epsilon parameter
        """
        x = torch.Tensor(x)
        with torch.no_grad():
            y0_pred, y1_pred, t_pred, eps = self.model(x)
        return y0_pred, y1_pred, t_pred, eps
