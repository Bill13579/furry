import time

class Hook:
    __slots__ = ["name", "history"]
    def __init__(self, name):
        self.name = name
        self.history = []
    
    def rec(self, val):
        self.history.append(val)

    @property
    def latest(self):
        return self.history[-1]

class HookAPI:
    class HookAlreadyExists(Exception): pass
    def __init__(self, name):
        self.name = name
        self.hooks = {}

    def new_hook(self, name, hook_cls=Hook):
        if name in self.hooks:
            raise HookAPI.HookAlreadyExists("hook `%s` already exists" % (name,))
        hook = hook_cls(name)
        self.hooks[name] = hook
        return hook
    
    def get_hook(self, name):
        return self.hooks[name]
    
    def hook(self, name):
        if name in self.hooks:
            return self.get_hook(name)
        hook = Hook(name)
        self.hooks[name] = hook
        return hook
