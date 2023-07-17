import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


class EarlyStop(Exception):
    """ Signal the end from iterator.__next__(). """
    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    value = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """generator return value"""


class TrainedModel(nn.Module):
    def __init__(
            self,
            model: nn.Module,
            criterion: nn.Module,
            optimizer: torch.optim.Optimizer,
            train_loader: DataLoader,
            valid_loader: DataLoader,
            test_loader: DataLoader,
            epochs: int,
            device: torch.device,
            dtype: torch.dtype,
            writer: SummaryWriter = SummaryWriter(),
            early_stop_patience: int = 400,
    ):
        super().__init__()
        self.epoch = None
        self.mean_train_loss = None
        self.mean_valid_loss = None
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.n_epochs = epochs
        self.device = device
        self.dtype = dtype
        self.writer = writer
        self.early_stop_patience = early_stop_patience

        self._early_stop_count = 0
        self._best_loss = math.inf

    def forward(self, X):
        self.model.forward(X)

    def _train_epoch(self):
        self.model.train()
        self.train_loss_record = []
        data_iter = iter(self.train_loader)

        while True:
            try:
                X, y = next(data_iter)
                X = X.to(device=self.device, dtype=self.dtype)
                y = y.to(device=self.device, dtype=self.dtype)

                pred = self.model.forward(X)
                loss = self.criterion(pred, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.train_loss_record.append(loss.detach().item())

            except StopIteration:
                break

    def after_train_epoch(self):
        mean_train_loss = sum(self.train_loss_record) / len(self.train_loss_record)
        self.writer.add_scalar("Loss/train", mean_train_loss)
        self.mean_train_loss = mean_train_loss

    def _valid_epoch(self):
        """evaluate the current model on validation set
        """
        self.model.eval()
        self.valid_loss_record = []
        data_iter = iter(self.data_iter)

        while True:
            try:
                X, y = next(data_iter)
                X = X.to(device=self.device, dtype=self.dtype)
                y = y.to(device=self.device, dtype=self.dtype)

                with torch.no_grad():
                    pred = self.model.forward(X)
                    loss = self.criterion(pred, y)

                self.valid_loss_record.append(loss.detach().item())

            except StopIteration:
                break

    def after_valid_epoch(self):
        mean_valid_loss = sum(self.valid_loss_record) / len(self.valid_loss_record)
        self.writer.add_scalar("Loss/valid", mean_valid_loss)
        self.mean_valid_loss = mean_valid_loss

    def fit(self):
        for epoch in range(self.n_epochs):
            self.epoch = epoch
            try:
                self._train_epoch()
                self.after_train_epoch()
                self._valid_epoch()
                self.after_valid_epoch()
                print(f'Epoch [{epoch + 1}/{self.n_epochs}]: Train loss: {self.mean_train_loss:.4f}, Valid loss: {self.mean_valid_loss:.4f}')
                self._early_stop_test()
            except EarlyStop:
                break

    def predict(self, X):
        self.model.eval()
        return self.model.forward(X)

    def _early_stop_test(self):
        if self.mean_valid_loss < self._best_loss:
            self._best_loss = self.mean_valid_loss
            self._early_stop_count = 0
        else:
            self._early_stop_count += 1

        if self._early_stop_count >= self.early_stop_patience:
            raise EarlyStop("Early stop at epoch {}".format(self.epoch))

    def save(self, filename):
        pass

# TODO: add the
