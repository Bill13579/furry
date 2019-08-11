import threading
from furry.data.data import Batch, Data

class ParallelLoadedData(Data):
    def load(self):
        self.loader = threading.Thread(target=self.load_data, args=self.__loader_args)
        self._start()
        self.__preload__(*self.__preload_args[0], *self.__preload_args[1])
        self.loader.start()

    def __init__(self, *args, cache=True, load_on_creation=False, **kwargs):
        super().__init__([], [])
        self.cache = cache
        self.current_x = []
        self.current_y = []
        self.current_metadata = []
        self.new_sample_loaded = threading.Event()
        self.finished = threading.Event()
        self.__stop = threading.Event()
        self.__stopped = threading.Event()
        self.__running = False
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
        self.__loader_args = (register_new_sample, finished)
        self.__preload_args = (args, kwargs)
        if load_on_creation:
            self.load()
    
    def __preload__(self, *args, **kwargs):
        pass

    @property
    def running(self):
        return self.__running

    def _stop(self):
        return self.__stop.is_set()

    def _start(self):
        self.__running = True
        self.__stopped.clear()

    def stop(self):
        self.__stop.set()
        self.__stopped.wait()
    
    def _stopped(self):
        self.__stop.clear()
        self.__stopped.set()
        self.__running = False

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
