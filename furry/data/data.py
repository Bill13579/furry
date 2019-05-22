from furry.data.utils import sync_shuffle

class Batch:
    __slots__ = ["x", "y", "metadata", "final"]
    def __init__(self, x, y, md, final=False):
        self.x = x
        self.y = y
        self.metadata = md
        self.final = final

class Data:
    def __init__(self, x, y, md=None):
        if len(x) != len(y) or (md is not None and len(x) != len(md)):
            raise ValueError("the length of x, y, and md must all be the same")
        self.__x = x
        self.__y = y
        if md is None:
            md = []
            for i in range(len(self.__x)):
                md.append({})
        self.__md = md
        self.__shuffled_x = self.__x[:]
        self.__shuffled_y = self.__y[:]
        self.__shuffled_md = self.__md[:]
        self.__i = 0
    
    @property
    def x(self):
        return self.__x
    
    @property
    def y(self):
        return self.__y
    
    @property
    def metadata(self):
        return self.__md
    
    @property
    def shuffled_x(self):
        return self.__shuffled_x
    
    @property
    def shuffled_y(self):
        return self.__shuffled_y
    
    @property
    def shuffled_metadata(self):
        return self.__shuffled_md
    
    @property
    def size(self):
        return len(self.x)

    def shuffle(self, seed=None):
        sync_shuffle(self.__shuffled_x, self.__shuffled_y, self.__shuffled_md, seed=seed)

    def append(self, x, y, md=None):
        if md is None:
            md = {}
        self.x.append(x)
        self.y.append(y)
        self.metadata.append(md)
        self.shuffled_x.append(x)
        self.shuffled_y.append(y)
        self.shuffled_metadata.append(md)
    
    def nbatch(self, batch_size=1):
        start, end = self.__i, self.__i + batch_size
        self.__i += batch_size
        done = self.__i >= self.size
        if done:
            self.__i = 0
        return Batch(self.shuffled_x[start:end], self.shuffled_y[start:end], self.shuffled_metadata[start:end], final=done)
    
    def generator(self, batch_size=1):
        batch = self.nbatch(batch_size=batch_size)
        yield batch
        while not batch.final:
            batch = self.nbatch(batch_size=batch_size)
            yield batch
