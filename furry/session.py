import math
import torch
from furry.logger import SessionLogger
from furry.data import sync_shuffle, upload, download
from furry.dev import default as default_device
from furry.loss import mse

class session:
    """Session class used for training a Furry model
    
    Attributes:
        model (furry.Module): The model to train.
        optimizer (furry.optimizer.Optimizer): Optimizer.
        loss (:obj:`function`, optional): The loss function to use. Defaults to `furry.loss.mse`.
        logger (:obj:`furry.logger.SessionLogger`, optional): Session logger. Defaults to `furry.logger.SessionLogger`.
        device (furry.device): The device to use. Defaults to `furry.dev.default`.
    """

    def __init__(self, model, optimizer, loss=mse, logger=SessionLogger(), dev=default_device):
        """Session class used for training a Furry model

        Args:
            model (furry.Module): The model to train.
            optimizer (furry.optimizer.Optimizer): Optimizer.
            loss (:obj:`function`, optional): The loss function to use. Defaults to `furry.loss.mse`.
            logger (:obj:`furry.logger.SessionLogger`, optional): Session logger. Defaults to `furry.logger.SessionLogger`.
            dev (:obj:`furry.device`, optional): The device to use. Defaults to `furry.dev.default`.
        """
        self.model = model
        self.optimizer = optimizer
        if self.optimizer.module is None:
            self.optimizer.module = self.model
        self.loss = loss
        self.logger = logger
        self.__dev = dev
    
    @property
    def device(self):
        """furry.device: The device used by this session"""
        return self.__dev

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass
    
    def fit(self, x, y, epochs=1, batch_size=32, shuffle=True):
        self.logger.new_session(epochs, batch_size)
        self.logger.stat.epoch_size = math.ceil(len(x) / batch_size)
        for i in range(epochs):
            self.logger.new_epoch()
            if shuffle:
                sync_shuffle(x, y)
            for xi, yi in zip(range(0,len(x),batch_size), range(0,len(y),batch_size)):
                xs = x[xi:xi+batch_size]
                ys = y[yi:yi+batch_size]
                xs = upload(torch.stack(xs), dev=self.device)
                ys = upload(torch.stack(ys), dev=self.device)
                self.logger.new_batch(xs, ys, len(xs))
                out = self.model.logits(xs)
                loss = self.loss(out, ys)
                loss.backward()
                self.optimizer.step()
                self.optimizer.reset_grads()
                self.logger.batch_end(download(loss).item(), out, xs, ys, len(xs))
                del xs, ys, out, loss
        self.logger.session_over()
    
    def fit_data(self, data, epochs=1, batch_size=32, shuffle=True):
        self.logger.new_session(epochs, batch_size)
        self.logger.stat.epoch_size = 1
        for i in range(epochs):
            self.logger.new_epoch()
            if shuffle:
                data.shuffle()
            for batch in data.generator(batch_size=batch_size):
                self.logger.stat.epoch_size = math.ceil(data.size / batch_size)
                xs = upload(torch.stack(batch.x), dev=self.device)
                ys = upload(torch.stack(batch.y), dev=self.device)
                self.logger.new_batch(xs, ys, len(xs))
                out = self.model.logits(xs)
                loss = self.loss(out, ys)
                loss.backward()
                self.optimizer.step()
                self.optimizer.reset_grads()
                self.logger.batch_end(download(loss).item(), out, xs, ys, len(xs))
                del xs, ys, out, loss, batch
        self.logger.session_over()
