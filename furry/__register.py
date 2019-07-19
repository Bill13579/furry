class Name:
    __slots__ = ["name", "i"]
    def __init__(self, name, i=None):
        self.name = name
        self.i = i
    
    def __str__(self):
        name = self.name
        if self.i is not None:
            name += "-%i" % (self.i,)
        return name

class Register:
    def __init__(self):
        self.names = {}
        self.counter = {}
    
    def register(self, module, name):
        self.names[module] = Name(name, self.counter.get(name))
        if name not in self.counter:
            self.counter[name] = 2
        else:
            self.counter[name] += 1
    
    def nameof(self, module):
        return str(self.names[module])

module_register = Register()
