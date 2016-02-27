# Define key bindings

from keys import key_patterns

class Bindings(object):
    def __init__(self, key_patterns):
        self._tag_to_key_labels = {}
        self._tag_to_help = {}
        self._key_label_to_tag = {}
        self._key_patterns = key_patterns
        self._cache_keycode_to_tag = {}
        
    def get_tag(self, keycode):
        '''Gets tag for keycode, returns None if no tag found.'''
        if keycode is None:
            return None
        if not keycode in self._cache_keycode_to_tag:
            label = self.get_key_label_from_keycode(keycode)
            self._cache_keycode_to_tag[keycode] = self.get_tag_from_key_label(label)
        return self._cache_keycode_to_tag[keycode]

    def get_tag_from_key_label(self, label):
        '''Get tag using key label, if no match, returns None.'''
        
        return self._key_label_to_tag.get(label, None)
        
    def get_key_label_from_keycode(self, keycode, extra_info = False):
        '''Get tag using keycode, if no match, returns None.'''
        
        label = None
        for mask in reversed(sorted(self._key_patterns.keys())):
            masked_keycode = keycode & mask
            if masked_keycode in self._key_patterns[mask]:
                label = self._key_patterns[mask][masked_keycode]
                break
        
        if extra_info:
            return label, [keycode & mask for mask in reversed(sorted(self._key_patterns.keys()))]
        else:
            return label
    
    def add(self, tag, key, help_text):
        self.add_multikey(tag, (key,), help_text)

    def add_multikey(self, tag, key_labels, help_text):
        for key_label in key_labels:
            assert key_label not in self._key_label_to_tag, (
                'Key "%s" cannot be bound to "%s" because it is already bound to "%s"' %
                (key_label, tag, self._key_label_to_tag[key_label])
                )
            self._key_label_to_tag[key_label] = tag
        self._tag_to_key_labels[tag] = key_labels
        self._tag_to_help[tag] = help_text

    def get_key_help(self, tag):
        return (self._tag_to_key_labels[tag], self._tag_to_help[tag])

_ = Bindings(key_patterns)

# Core
_.add('freeze_cam', 'f',
       'Freeze or unfreeze camera capture')
_.add('toggle_input_mode', 'c',
       'Toggle between camera and static files')
_.add('static_file_increment', 'e',
       'Load next static file')
_.add('static_file_decrement', 'w',
       'Load previous static file')
_.add('help_mode', 'h',
       'Toggle this help screen')
_.add('stretch_mode', '0',
       'Toggle between cropping and stretching static files to be square')
_.add('debug_level', '5',
       'Cycle debug level between 0 (quiet), 1 (some timing info) and 2 (all timing info)')
_.add('quit', 'q',
       'Quit')

# Caffevis
_.add_multikey('reset_state', ['esc'],
       'Reset: turn off backprop, reset to layer 0, unit 0, default boost.')
_.add_multikey('sel_left', ['left', 'j'],
       '')
_.add_multikey('sel_right', ['right', 'l'],
       '')
_.add_multikey('sel_down', ['down', 'k'],
       '')
_.add_multikey('sel_up', ['up', 'i'],
       '')
_.add('sel_left_fast', 'J',
       '')
_.add('sel_right_fast', 'L',
       '')
_.add('sel_down_fast', 'K',
       '')
_.add('sel_up_fast', 'I',
       '')
_.add_multikey('sel_layer_left', ['u', 'U'],
       'Select previous layer without moving cursor')
_.add_multikey('sel_layer_right', ['o', 'O'],
       'Select next layer without moving cursor')

_.add('zoom_mode', 'z',
       'Cycle zooming through {currently selected unit, backprop results, none}')
_.add('pattern_mode', 's',
       'Toggle overlay of preferred input pattern (regularized optimized images)')

_.add('ez_back_mode_loop', 'b',
       'Cycle through a few common backprop/deconv modes')
_.add('freeze_back_unit', 'd',
       'Freeze the bprop/deconv origin to be the currently selected unit')
_.add('show_back', 'a',
       'Toggle between showing forward activations and back/deconv diffs')
_.add('back_mode', 'n',
       '(expert) Change back mode directly.')
_.add('back_filt_mode', 'm',
       '(expert) Change back output filter directly.')

_.add('boost_gamma', 't',
       'Boost contrast using gamma correction')
_.add('boost_individual', 'T',
       'Boost contrast by scaling each channel to use more of its individual range')
_.add('toggle_label_predictions', '8',
       'Turn on or off display of prob label values')
_.add('toggle_unit_jpgs', '9',
       'Turn on or off display of loaded jpg visualization')

bindings = _
