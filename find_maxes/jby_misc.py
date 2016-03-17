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
            titlestr = self.title + ' ' if self.title else ''
            print 'Elapsed %sreal: %.06f, sys: %.06f' % ((titlestr,) + self.elapsed())

def misc_main():
    print 'Running quick demo'

    print ' -> Before with statement'
    with WithTimer('sleepy time'):
        print '   -> About to sleep'
        time.sleep(.5)
        print '   -> Done sleeping'
    print ' -> After with statement'
        
    
if __name__ == '__main__':
    misc_main()
