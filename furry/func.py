from furry.module import Module

class Func(Module):
    def __init__(self, f, f2=None, name="Func", dev=None):
        super().__init__(name=name, dev=dev)
        self.__f = f
        self.__f2 = f2
    
    @property
    def f(self):
        return self.__f
    
    @property
    def f2(self):
        return self.__f2 if self.__f2 is not None else self.__f
    
    def __broadcast__(self, x):
        return x

    def __logits__(self, x, *args, **kwargs):
        return self.f(x, *args, **kwargs)
    
    def __forward__(self, x, *args, **kwargs):
        return self.f2(x, *args, **kwargs)

def __cfunc_module(f, n):
    class cfunc(Func):
        def __init__(self, name=n, dev=None):
            super().__init__(f, name=name, dev=dev)
    return cfunc
