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
       'Toggle between cropping and stretching static files to be square')
_.add('debug_level', Key.n5,
       'Cycle debug level between 0 (quiet), 1 (some timing info) and 2 (all timing info)')
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
       'Select previous layer without moving cursor')
_.add_multikey('sel_layer_right', [Key.o,Key.O],
       'Select next layer without moving cursor')

_.add('zoom_mode', Key.z,
       'Cycle zooming through {currently selected unit, backprop results, none}')
_.add('pattern_mode', Key.s,
       'Toggle overlay of preferred input stimulus (regularized optimized images)')

_.add('ez_back_mode_loop', Key.b,
       'Cycle through a few common backprop/deconv modes')
_.add('freeze_back_unit', Key.d,
       'Freeze the bprop/deconv origin to be the currently selected unit')
_.add('show_back', Key.a,
       'Toggle between showing forward activations and back/deconv diffs')
_.add('back_mode', Key.n,
       '(expert) Change back mode directly.')
_.add('back_filt_mode', Key.m,
       '(expert) Change back output filter directly.')

_.add('boost_gamma', Key.t,
       'Boost contrast using gamma correction')
_.add('boost_individual', Key.T,
       'Boost contrast by scaling each channel to use more of its individual range')
_.add('toggle_label_predictions', Key.n8,
       'Turn on or off display of prob label values')
_.add('toggle_unit_jpgs', Key.n9,
       'Turn on or off display of loaded jpg visualization')

bindings = _
