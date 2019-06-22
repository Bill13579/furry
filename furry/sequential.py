import furry.data
import furry.module
from furry.__register import module_register

class Sequential(furry.module.Module):
    def __init__(self, *layers, dtype=furry.data.float32, name=None, dev=None):
        super().__init__(input_rank=layers[0].input_rank, dtype=dtype, name=name, dev=dev)
        self.__layers = list(layers)
        for l in self.layers:
            self.add_module(module_register.nameof(l), l)
    
    def add(self, layer):
        self.layers.append(layer)
        self.add_module(module_register.nameof(layer), layer)

    @property
    def layers(self):
        return self.__layers
    
    def __logits__(self, x):
        out = x
        for l in self.layers:
            out = l.logits(out)
        return out
    
    def __forward__(self, x):
        out = x
        for l in self.layers:
            out = l.forward(out)
        return out
