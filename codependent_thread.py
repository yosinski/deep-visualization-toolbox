import time
from threading import Lock, Thread


class CodependentThread(Thread):
    '''A Thread that must be occasionally poked to stay alive. '''

    def __init__(self, heartbeat_timeout = 1.0):
        Thread.__init__(self)
        self.heartbeat_timeout = heartbeat_timeout
        self.heartbeat_lock = Lock()
        self.heartbeat()
        
    def heartbeat(self):
        with self.heartbeat_lock:
            self.last_beat = time.time()

    def is_timed_out(self):
        with self.heartbeat_lock:
            now = time.time()
            if now - self.last_beat > self.heartbeat_timeout:
                print '%s instance %s timed out after %s seconds (%s - %s = %s)' % (self.__class__.__name__, self, self.heartbeat_timeout, now, self.last_beat, now - self.last_beat)
                return True
            else:
                return False
