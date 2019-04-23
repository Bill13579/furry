import threading
from furry.data.data import Batch, Data

class ParallelLoadedData(Data):
    def __init__(self):
        super().__init__([], [])
        self.current_x = []
        self.current_y = []
        self.new_sample_loaded = threading.Event()
        self.finished = threading.Event()
        def register_new_sample(x, y):
            self.append(x, y)
            self.current_x.append(x)
            self.current_y.append(y)
            self.new_sample_loaded.set()
            self.new_sample_loaded.clear()
        def finished():
            self.finished.set()
        self.loader = threading.Thread(target=self.load_data, args=(register_new_sample, finished))
        self.loader.start()

    def load_data(self, register_new_sample, finished):
        pass

    def nbatch(self, batch_size=1):
        final = False
        if len(self.current_x) < batch_size:
            for i in range(batch_size - len(self.current_x)):
                if self.finished.is_set():
                    final = True
                self.new_sample_loaded.wait()
        batch = Batch(self.current_x[:batch_size], self.current_y[:batch_size], final=final)
        del self.current_x[:batch_size]
        del self.current_y[:batch_size]
        return batch

class OnTheFlyData(Data):
    def __init__(self):
        super().__init__([None], [None])
    
    def __has_ntei(self):
        return (self.x[-1] is None) and (self.y[-1] is None) and (self.shuffled_x[-1] is None) and (self.shuffled_y[-1] is None)

    def __remove_ntei_element(self, arr):
        if arr[-1] is None:
            del arr[-1]

    def __remove_ntei(self):
        self.__remove_ntei_element(self.x)
        self.__remove_ntei_element(self.y)
        self.__remove_ntei_element(self.shuffled_x)
        self.__remove_ntei_element(self.shuffled_y)

    def single(self):
        raise NotImplementedError()
    
    def nbatch(self, batch_size=1):
        if self.__has_ntei():
            self.__remove_ntei()
            for i in range(batch_size):
                dp = self.single()
                if "x" not in dp or "y" not in dp:
                    raise ValueError("single() must return a dict with keys x and y")
                self.append(dp["x"], dp["y"])
                if "done" in dp and dp["done"]:
                    self.__remove_ntei()
                    break
            self.append(None, None)
        return super().nbatch(batch_size=batch_size)
