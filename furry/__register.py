class Register:
    def __init__(self):
        self.names = {}
    
    def register(self, module, name):
        self.names[module] = name
    
    def nameof(self, module):
        return self.names[module]

module_register = Register()
