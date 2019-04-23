from furry.data.utils import sync_shuffle

class Batch:
    __slots__ = ["x", "y", "final"]
    def __init__(self, x, y, final=False):
        self.x = x
        self.y = y
        self.final = final

class Data:
    def __init__(self, x, y):
        if len(x) != len(y):
            raise ValueError("the length of x must equal the length of y")
        self.__x = x
        self.__y = y
        self.__shuffled_x = self.__x[:]
        self.__shuffled_y = self.__y[:]
        self.__i = 0
        self.shuffle()
    
    @property
    def x(self):
        return self.__x
    
    @property
    def y(self):
        return self.__y
    
    @property
    def shuffled_x(self):
        return self.__shuffled_x
    
    @property
    def shuffled_y(self):
        return self.__shuffled_y
    
    @property
    def size(self):
        return len(self.x)

    def shuffle(self, seed=None):
        sync_shuffle(self.__shuffled_x, self.__shuffled_y, seed=seed)

    def append(self, x, y):
        self.x.append(x)
        self.y.append(y)
        self.shuffled_x.append(x)
        self.shuffled_y.append(y)
    
    def nbatch(self, batch_size=1):
        start, end = self.__i, self.__i + batch_size
        batch_x = self.x[start:end]
        batch_y = self.y[start:end]
        self.__i += batch_size
        done = False
        if self.__i >= self.size:
            done = True
            self.__i = 0
        return Batch(batch_x, batch_y, final=done)
    
    def generator(self, batch_size=1):
        batch = self.nbatch(batch_size=batch_size)
        yield batch
        while not batch.final:
            batch = self.nbatch(batch_size=batch_size)
            yield batch
