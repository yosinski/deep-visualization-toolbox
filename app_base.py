#! /usr/bin/env python

class BaseApp(object):
    '''Base App class.'''

    def __init__(self, settings, key_bindings):
        self.debug_level = 0

    def handle_input(self, input_image, panes):
        pass
        
    def handle_key(self, key, panes):
        '''Handle key and return either key (to let someone downstream handle it) or None (if this app handled it)'''
        pass

    def redraw_needed(self, key, panes):
        '''App should return whether or not its internal state has
        been updated (perhaps in response to handle_key, handle_input,
        or some internal processing finishing).
        '''
        return False

    def draw(self, panes):
        '''Tells the app to draw in the given panes. Returns True if panes were changed and require a redraw, False if nothing was changed.'''
        return False

    def draw_help(self, panes):
        '''Tells the app to draw its help screen in the given pane. No return necessary.'''
        pass

    def start(self):
        '''Notify app to start, possibly creating any necessary threads'''
        pass

    def get_heartbeats(self):
        '''Returns a list of heartbeat functions, if any, that should be called regularly.'''
        return []

    def set_debug(self, level):
        self.debug_level = level

    def quit(self):
        '''Notify app to quit, possibly joining any threads'''
        pass
