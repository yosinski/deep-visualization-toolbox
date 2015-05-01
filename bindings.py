# Define key bindings

from keys import Key

class Bindings(object):
    def __init__(self):
        self._tag2keycodes = {}
        self._tag2keystrings = {}
        self._tag2help = {}
        self._keys_seen = set()
        
    def match(self, tag, key):
        return key in self._tag2keycodes[tag]
        
    def add(self, tag, key, help_text):
        self.add_multikey(tag, (key,), help_text)

    def add_multikey(self, tag, keys, help_text):
        for key in keys:
            assert key not in self._keys_seen, ('Key %s is already bound' % key[1])
            self._keys_seen.add(key)
        self._tag2keycodes[tag]   = tuple([key[0] for key in keys])
        self._tag2keystrings[tag] = tuple([key[1] for key in keys])
        self._tag2help[tag] = help_text

    def get_key_help(self, tag):
        return (self._tag2keystrings[tag], self._tag2help[tag])

_ = Bindings()

# Core
_.add('freeze_cam', Key.f,
       'Freeze or unfreeze camera capture')
_.add('toggle_input_mode', Key.c,
       'Toggle between camera and static files')
_.add('static_file_increment', Key.e,
       'Load next static file')
_.add('static_file_decrement', Key.w,
       'Load previous static file')
_.add('help_mode', Key.h,
       'Toggle this help screen')
_.add('stretch_mode', Key.n0,
       'Toggle between cropping/stretching static files to square')
_.add('debug_level', Key.n5,
       'Toggle debug level between 0 (quiet), 1 (some timing info) and 2 (all timing info)')
_.add('quit', Key.q,
       'Quit')

# Caffevis
_.add_multikey('reset_state', [Key.esc],
       'Reset: turn off backprop, reset to layer 0, unit 0, default boost.')
_.add_multikey('sel_left', [Key.left,Key.j],
       '')
_.add_multikey('sel_right', [Key.right,Key.l],
       '')
_.add_multikey('sel_down', [Key.down,Key.k],
       '')
_.add_multikey('sel_up', [Key.up,Key.i],
       '')
_.add('sel_left_fast', Key.J,
       '')
_.add('sel_right_fast', Key.L,
       '')
_.add('sel_down_fast', Key.K,
       '')
_.add('sel_up_fast', Key.I,
       '')
_.add_multikey('sel_layer_left', [Key.u,Key.U],
       '')
_.add_multikey('sel_layer_right', [Key.o,Key.O],
       '')
_.add('boost_gamma', Key.t,
       '')
_.add('boost_individual', Key.T,
       '')
_.add('pattern_mode', Key.s,
       'Show or hide overlay of preferred input image (found via regularized optimization)')
_.add('show_back', Key.a,
       'Toggle showing forward pass activations or backward pass activations')
_.add('ez_back_mode_loop', Key.b,
       '')
_.add('back_mode', Key.n,
       '')
_.add('back_filt_mode', Key.m,
       '')
_.add('freeze_back_unit', Key.d,
       '')
_.add('zoom_mode', Key.z,
       '')
_.add('toggle_label_predictions', Key.n8,
       'Turn on or off display of prob label values')
_.add('toggle_unit_jpgs', Key.n9,
       'Turn on or off display of loaded jpg visualization')

bindings = _
