import torch
import furry.utils
from furry.dev import default as default_device
from furry.data.utils import float32
from furry.__register import module_register

class Module(torch.nn.Module):
    def __init__(self, input_rank=None, dtype=float32, name=None, dev=None):
        super(Module, self).__init__()
        if dev is None:
            dev = default_device
        self.__input_rank = input_rank
        self.__dtype = dtype
        module_register.register(self, name)
        self.__dev = dev
        self.__init_done = False

    @property
    def input_rank(self):
        return self.__input_rank

    @property
    def dtype(self):
        return self.__dtype
    
    @property
    def name(self):
        return module_register.nameof(self)
    
    @property
    def device(self):
        return self.__dev
    
    def _init_done(self):
        self.to(self.device)
        self.__init_done = True
    
    def init(self, input_size):
        pass
    
    def __forward__(self, x, **kwargs):
        return self.logits(x, **kwargs)
    
    def forward(self, x, **kwargs):
        return self.__forward__(x, **kwargs)
    
    def __broadcast__(self, x):
        if len(x.size()) == self.__input_rank:
            x = furry.data.prepend_dimension(x)
        return x
    
    def __logits__(self, x):
        return x
    
    def logits(self, x, **kwargs):
        x = self.__broadcast__(x)
        if not self.__init_done:
            self.init(x.size()[1:])
        return self.__logits__(x, **kwargs)
    
    def sv(self, path):
        Module.save(self, path)
    
    def ld(self, path):
        Module.load(self, path, dev=self.__dev)
    
    @staticmethod
    def save(module, path):
        torch.save(module.state_dict(), path)
    
    @staticmethod
    def load(module, path, dev=default_device):
        module.load_state_dict(torch.load(path, map_location=dev))
