import sys

def strblen(text):
    return len(text.encode('utf-8'))

class Logger:
    def write(self, text, end='', file=sys.stdout, flush=True):
        file.write(text + end)
        if flush:
            file.flush()
        self.bytes_written = strblen(text + end)
    
    def inline_write(self, text, end='', file=sys.stdout, flush=True):
        bytes_written = strblen(text + end)
        self.write("\r" + text + end + (max(self.bytes_written - bytes_written, 0)) * " ", end='', file=file, flush=flush)
        self.bytes_written = bytes_written

class TrainingLogger(Logger):
    class Session:
        def __init__(self, epochs, epoch_size, batch_size):
            self.sample = 0
            self.batch = 0
            self.epoch = 0
            self.epochs = epochs
            self.epoch_size = epoch_size
            self.batch_size = batch_size
            self.over = False
    
    def new_session(self, epochs, epoch_size, batch_size):
        self.stat = self.Session(epochs, epoch_size, batch_size)
    
    def new_epoch(self):
        self.stat.sample = 0
        self.stat.batch = 0
        self.stat.epoch += 1
    
    def new_batch(self, batch_size):
        self.stat.sample = 0
        self.stat.batch += 1
    
    def batch_end(self, batch_size, model_loss):
        pass
    
    def new_sample(self, sample, model_output, model_loss):
        self.stat.sample += 1
    
    def session_over(self):
        self.stat.over = True
