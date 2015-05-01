#! /usr/bin/env python


import time


class WithTimer:
    def __init__(self, title = '', quiet = False):
        self.title = title
        self.quiet = quiet
        
    def elapsed(self):
        return time.time() - self.wall, time.clock() - self.proc

    def enter(self):
        '''Manually trigger enter'''
        self.__enter__()
    
    def __enter__(self):
        self.proc = time.clock()
        self.wall = time.time()
        return self
        
    def __exit__(self, *args):
        if not self.quiet:
            titlestr = (' ' + self.title) if self.title else ''
            print 'Elapsed%s: wall: %.06f, sys: %.06f' % ((titlestr,) + self.elapsed())
