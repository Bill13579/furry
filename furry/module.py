import torch
import furry.utils
from furry.utils import default_device

float32 = torch.float32
float64 = torch.float64
float16 = torch.float16
uint8 = torch.uint8
int8 = torch.int8
int16 = torch.int16
int32 = torch.int32
int64 = torch.int64

class Module(torch.nn.Module):
    def __init__(self, input_rank=None, dtype=float32, dev=None):
        super(Module, self).__init__()
        if dev is None:
            dev = default_device
        self._input_rank = input_rank
        self._dtype = dtype
        self._dev = dev
        self.__init_done = False
    
    @property
    def dtype(self):
        return self._dtype
    
    @property
    def device(self):
        return self._dev
    
    def _init_done(self):
        self.to(self.device)
        self.__init_done = True
    
    def init(self, input_size):
        pass
    
    def __forward__(self, x):
        return x
    
    def forward(self, x, **kwargs):
        x = self.logits(x, **kwargs)
        return self.__forward__(x)
    
    def __broadcast__(self, x):
        if len(x.size()) == self._input_rank:
            x = furry.utils.add_batch_dimension(x)
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
        Module.load(self, path, dev=self._dev)
    
    @staticmethod
    def save(module, path):
        torch.save(module.state_dict(), path)
    
    @staticmethod
    def load(module, path, dev=default_device):
        module.load_state_dict(torch.load(path, map_location=dev))
