import numpy as np
from collections import OrderedDict
from threading import RLock

class FIFOLimitedArrayCache(object):
    '''Threadsafe cache that stores numpy arrays (or any other object that
    defines obj.nbytes) and limits total memory. Items are ejected, if
    necessary, in the same order in which they were added.
    '''

    def __init__(self, max_bytes = 1e7):
        self._store = OrderedDict()
        self._store_bytes = 0
        self._max_bytes = max_bytes
        self._lock = RLock()
        
    def get(self, key, default = None):
        with self._lock:
            if key in self._store:
                return self._store[key]
            else:
                return default

    def set(self, key, val):
        with self._lock:
            if key in self._store:
                self._store_bytes -= self._store[key].nbytes
            self._store[key] = val
            self._store_bytes += self._store[key].nbytes
            self._trim()

    def _trim(self):
        while len(self._store) > 0 and self._store_bytes > self._max_bytes:
            key,val = self._store.popitem(last = False)
            self._store_bytes -= val.nbytes
        
    def delete(self, key, raise_if_missing = False):
        with self._lock:
            if key in self._store:
                self._store_bytes -= val.nbytes
                del self._store[key]
            elif raise_if_missing:
                raise Exception('key %s not found in cache' % repr(key))

    def get_size(self):
        return self._store_bytes

    def __str__(self):
        with self._lock:
            return 'FIFOLimitedArrayCache<%d items, bytes used/max %g/%g >' % (len(self._store), self._store_bytes, self._max_bytes)
