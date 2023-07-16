import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class Training(object):
    def __init__(
            self,
            model: nn.Module,
            criterion: nn.Module,
            optimizer: torch.optim.Optimizer,
            train_loader: DataLoader,
            valid_loader: DataLoader,
            test_loader: DataLoader,
            epochs: int,
            device: torch.device = torch.device("cpu"),
            dtype: torch.dtype = torch.float64,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.device = device
        self.dtype = dtype

    def _train_epoch(self):
        self.model.train()
        data_iter = iter(self.train_loader)

        while True:
            try:
                X_train, y_train = next(data_iter)
                X_train = X_train.to(device=self.device, dtype=self.dtype)
                y_train = y_train.to(device=self.device, dtype=self.dtype)

                y_pred = self.model.forward(X_train)
                loss = self.criterion(y_pred, y_train)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            except StopIteration:
                break

    def _valid_epoch(self):
        """evaluate the current model on validation set
        """
        self.model.eval()
        data_iter = iter(self.valid_loader)

        while True:
            try:
                X_valid, y_valid = next(data_iter)
                X_valid = X_valid.to(device=self.device, dtype=self.dtype)
                y_valid = y_valid.to(device=self.device, dtype=self.dtype)

                with torch.no_grad():
                    y_pred = self.model.forward(X_valid)

                loss = self.criterion(y_pred, y_valid)
                # do something with the loss

            except StopIteration:
                break

    def fit(self):
        for i in range(self.epochs):
            self._train_epoch()
            self._valid_epoch()

    def predict(self, X):
        self.model.eval()
        return self.model.forward(X)

    def save(self, filename):
        pass


# TODO: add the