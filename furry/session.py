import math
import torch
from furry.logger import TrainingLogger
from furry.utils import shuffle_data, upload, download
from furry.loss import mse

class session:
    def __init__(self, model, optimizer, loss=mse, logger=TrainingLogger()):
        self.model = model
        self.optimizer = optimizer
        if self.optimizer.module is None:
            self.optimizer.module = self.model
            self.optimizer.init()
        self.loss = loss
        self.logger = logger

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass
    
    def fit(self, x, y, epochs=1, batch_size=32, shuffle=True):
        self.logger.new_session(epochs, math.ceil(len(x) / batch_size), batch_size)
        for i in range(epochs):
            self.logger.new_epoch()
            if shuffle:
                shuffle_data(x, y)
            for xi, yi in zip(range(0,len(x),batch_size), range(0,len(y),batch_size)):
                xs = x[xi:xi+batch_size]
                ys = y[yi:yi+batch_size]
                xs = upload(torch.stack(xs))
                ys = upload(torch.stack(ys))
                self.logger.new_batch(len(xs))
                out = self.model(xs)
                loss = self.loss(out, ys)
                print(loss)
                loss.backward()
                self.optimizer.step()
                self.optimizer.reset_grads()
                self.logger.batch_end(len(xs), download(loss).item())
                del xs, ys, out, loss
        self.logger.session_over()
