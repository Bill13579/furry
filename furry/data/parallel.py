import threading
from furry.data.data import Batch, Data

class ParallelLoadedData(Data):
    def load(self):
        self.loader.start()

    def __init__(self, *args, cache=True, **kwargs):
        super().__init__([], [])
        self.cache = cache
        self.current_x = []
        self.current_y = []
        self.current_metadata = []
        self.new_sample_loaded = threading.Event()
        self.finished = threading.Event()
        self.__finished_or_new_sample_loaded = threading.Event()
        def register_new_sample(x, y, md=None):
            if md is None:
                md = {}
            if self.cache:
                self.append(x, y, md=md)
            self.current_x.append(x)
            self.current_y.append(y)
            self.current_metadata.append(md)
            self.new_sample_loaded.set()
            self.new_sample_loaded.clear()
            self.__finished_or_new_sample_loaded.set()
            self.__finished_or_new_sample_loaded.clear()
        def finished():
            self.finished.set()
            self.__finished_or_new_sample_loaded.set()
        self.loader = threading.Thread(target=self.load_data, args=(register_new_sample, finished))
        self.__preload__(*args, **kwargs)
        self.load()
    
    def __preload__(self, *args, **kwargs):
        pass

    def load_data(self, register_new_sample, finished):
        pass

    def nbatch(self, batch_size=1, as_tensor=False):
        if not self.finished.is_set():
            self._Data__i += batch_size
            final = False
            if len(self.current_x) < batch_size:
                for i in range(batch_size - len(self.current_x)):
                    self.__finished_or_new_sample_loaded.wait()
                    if self.finished.is_set():
                        final = True
                        self._Data__i = 0
                        break
            batch = Batch(self.current_x[:batch_size], self.current_y[:batch_size], self.current_metadata[:batch_size], final=final)
            del self.current_x[:batch_size]
            del self.current_y[:batch_size]
            del self.current_metadata[:batch_size]
            self._nbatch_batch_type_set(batch, as_tensor)
        else:
            if self.cache:
                batch = super().nbatch(batch_size=batch_size)
            else:
                batch = Batch(None, None, None, final=True)
        return batch
