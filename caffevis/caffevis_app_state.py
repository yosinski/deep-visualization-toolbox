import time
from threading import Lock



class CaffeVisAppState(object):
    '''State of CaffeVis app.'''

    def __init__(self, net, settings, bindings, net_layer_info):
        self.lock = Lock()  # State is accessed in multiple threads
        self.settings = settings
        self.bindings = bindings
        self._layers = net.blobs.keys()
        self._layers = self._layers[1:]  # chop off data layer
        if hasattr(self.settings, 'caffevis_filter_layers'):
            for name in self._layers:
                if self.settings.caffevis_filter_layers(name):
                    print '  Layer filtered out by caffevis_filter_layers: %s' % name
            self._layers = filter(lambda name: not self.settings.caffevis_filter_layers(name), self._layers)
        self.net_layer_info = net_layer_info
        self.layer_boost_indiv_choices = self.settings.caffevis_boost_indiv_choices   # 0-1, 0 is noop
        self.layer_boost_gamma_choices = self.settings.caffevis_boost_gamma_choices   # 0-inf, 1 is noop
        self.caffe_net_state = 'free'     # 'free', 'proc', or 'draw'
        self.extra_msg = ''
        self.back_stale = True       # back becomes stale whenever the last back diffs were not computed using the current backprop unit and method (bprop or deconv)
        self.next_frame = None
        self.jpgvis_to_load_key = None
        self.last_key_at = 0
        self.quit = False

        self._reset_user_state()

    def _reset_user_state(self):
        self.layer_idx = 0
        self.layer = self._layers[0]
        self.layer_boost_indiv_idx = self.settings.caffevis_boost_indiv_default_idx
        self.layer_boost_indiv = self.layer_boost_indiv_choices[self.layer_boost_indiv_idx]
        self.layer_boost_gamma_idx = self.settings.caffevis_boost_gamma_default_idx
        self.layer_boost_gamma = self.layer_boost_gamma_choices[self.layer_boost_gamma_idx]
        self.cursor_area = 'top'   # 'top' or 'bottom'
        self.selected_unit = 0
        # Which layer and unit (or channel) to use for backprop
        self.backprop_layer = self.layer
        self.backprop_unit = self.selected_unit
        self.backprop_selection_frozen = False    # If false, backprop unit tracks selected unit
        self.back_enabled = False
        self.back_mode = 'grad'      # 'grad' or 'deconv'
        self.back_filt_mode = 'raw'  # 'raw', 'gray', 'norm', 'normblur'
        self.pattern_mode = False    # Whether or not to show desired patterns instead of activations in layers pane
        self.layers_pane_zoom_mode = 0       # 0: off, 1: zoom selected (and show pref in small pane), 2: zoom backprop
        self.layers_show_back = False   # False: show forward activations. True: show backward diffs
        self.show_label_predictions = self.settings.caffevis_init_show_label_predictions
        self.show_unit_jpgs = self.settings.caffevis_init_show_unit_jpgs
        self.drawing_stale = True
        kh,_ = self.bindings.get_key_help('help_mode')
        self.extra_msg = '%s for help' % kh[0]
        
    def handle_key(self, key):
        #print 'Ignoring key:', key
        if key == -1:
            return key

        with self.lock:
            key_handled = True
            self.last_key_at = time.time()
            tag = self.bindings.get_tag(key)
            if tag == 'reset_state':
                self._reset_user_state()
            elif tag == 'sel_layer_left':
                #hh,ww = self.tiles_height_width
                #self.selected_unit = self.selected_unit % ww   # equivalent to scrolling all the way to the top row
                #self.cursor_area = 'top' # Then to the control pane
                self.layer_idx = max(0, self.layer_idx - 1)
                self.layer = self._layers[self.layer_idx]
            elif tag == 'sel_layer_right':
                #hh,ww = self.tiles_height_width
                #self.selected_unit = self.selected_unit % ww   # equivalent to scrolling all the way to the top row
                #self.cursor_area = 'top' # Then to the control pane
                self.layer_idx = min(len(self._layers) - 1, self.layer_idx + 1)
                self.layer = self._layers[self.layer_idx]

            elif tag == 'sel_left':
                self.move_selection('left')
            elif tag == 'sel_right':
                self.move_selection('right')
            elif tag == 'sel_down':
                self.move_selection('down')
            elif tag == 'sel_up':
                self.move_selection('up')

            elif tag == 'sel_left_fast':
                self.move_selection('left', self.settings.caffevis_fast_move_dist)
            elif tag == 'sel_right_fast':
                self.move_selection('right', self.settings.caffevis_fast_move_dist)
            elif tag == 'sel_down_fast':
                self.move_selection('down', self.settings.caffevis_fast_move_dist)
            elif tag == 'sel_up_fast':
                self.move_selection('up', self.settings.caffevis_fast_move_dist)

            elif tag == 'boost_individual':
                self.layer_boost_indiv_idx = (self.layer_boost_indiv_idx + 1) % len(self.layer_boost_indiv_choices)
                self.layer_boost_indiv = self.layer_boost_indiv_choices[self.layer_boost_indiv_idx]
            elif tag == 'boost_gamma':
                self.layer_boost_gamma_idx = (self.layer_boost_gamma_idx + 1) % len(self.layer_boost_gamma_choices)
                self.layer_boost_gamma = self.layer_boost_gamma_choices[self.layer_boost_gamma_idx]
            elif tag == 'pattern_mode':
                self.pattern_mode = not self.pattern_mode
                if self.pattern_mode and not hasattr(self.settings, 'caffevis_unit_jpg_dir'):
                    print 'Cannot switch to pattern mode; caffevis_unit_jpg_dir not defined in settings.py.'
                    self.pattern_mode = False
            elif tag == 'show_back':
                # If in pattern mode: switch to fwd/back. Else toggle fwd/back mode
                if self.pattern_mode:
                    self.pattern_mode = False
                else:
                    self.layers_show_back = not self.layers_show_back
                if self.layers_show_back:
                    if not self.back_enabled:
                        self.back_enabled = True
                        self.back_stale = True
            elif tag == 'back_mode':
                if not self.back_enabled:
                    self.back_enabled = True
                    self.back_mode = 'grad'
                    self.back_stale = True
                else:
                    if self.back_mode == 'grad':
                        self.back_mode = 'deconv'
                        self.back_stale = True
                    else:
                        self.back_enabled = False
            elif tag == 'back_filt_mode':
                    if self.back_filt_mode == 'raw':
                        self.back_filt_mode = 'gray'
                    elif self.back_filt_mode == 'gray':
                        self.back_filt_mode = 'norm'
                    elif self.back_filt_mode == 'norm':
                        self.back_filt_mode = 'normblur'
                    else:
                        self.back_filt_mode = 'raw'
            elif tag == 'ez_back_mode_loop':
                # Cycle:
                # off -> grad (raw) -> grad(gray) -> grad(norm) -> grad(normblur) -> deconv
                if not self.back_enabled:
                    self.back_enabled = True
                    self.back_mode = 'grad'
                    self.back_filt_mode = 'raw'
                    self.back_stale = True
                elif self.back_mode == 'grad' and self.back_filt_mode == 'raw':
                    self.back_filt_mode = 'norm'
                elif self.back_mode == 'grad' and self.back_filt_mode == 'norm':
                    self.back_mode = 'deconv'
                    self.back_filt_mode = 'raw'
                    self.back_stale = True
                else:
                    self.back_enabled = False
            elif tag == 'freeze_back_unit':
                # Freeze selected layer/unit as backprop unit
                self.backprop_selection_frozen = not self.backprop_selection_frozen
                if self.backprop_selection_frozen:
                    # Grap layer/selected_unit upon transition from non-frozen -> frozen
                    self.backprop_layer = self.layer
                    self.backprop_unit = self.selected_unit                    
            elif tag == 'zoom_mode':
                self.layers_pane_zoom_mode = (self.layers_pane_zoom_mode + 1) % 3
                if self.layers_pane_zoom_mode == 2 and not self.back_enabled:
                    # Skip zoom into backprop pane when backprop is off
                    self.layers_pane_zoom_mode = 0

            elif tag == 'toggle_label_predictions':
                self.show_label_predictions = not self.show_label_predictions

            elif tag == 'toggle_unit_jpgs':
                self.show_unit_jpgs = not self.show_unit_jpgs

            else:
                key_handled = False

            self._ensure_valid_selected()

            self.drawing_stale = key_handled   # Request redraw any time we handled the key

        return (None if key_handled else key)

    def redraw_needed(self):
        with self.lock:
            return self.drawing_stale

    def move_selection(self, direction, dist = 1):
        if direction == 'left':
            if self.cursor_area == 'top':
                self.layer_idx = max(0, self.layer_idx - dist)
                self.layer = self._layers[self.layer_idx]
            else:
                self.selected_unit -= dist
        elif direction == 'right':
            if self.cursor_area == 'top':
                self.layer_idx = min(len(self._layers) - 1, self.layer_idx + dist)
                self.layer = self._layers[self.layer_idx]
            else:
                self.selected_unit += dist
        elif direction == 'down':
            if self.cursor_area == 'top':
                self.cursor_area = 'bottom'
            else:
                self.selected_unit += self.net_layer_info[self.layer]['tile_cols'] * dist
        elif direction == 'up':
            if self.cursor_area == 'top':
                pass
            else:
                self.selected_unit -= self.net_layer_info[self.layer]['tile_cols'] * dist
                if self.selected_unit < 0:
                    self.selected_unit += self.net_layer_info[self.layer]['tile_cols']
                    self.cursor_area = 'top'

    def _ensure_valid_selected(self):
        n_tiles = self.net_layer_info[self.layer]['n_tiles']

        # Forward selection
        self.selected_unit = max(0, self.selected_unit)
        self.selected_unit = min(n_tiles-1, self.selected_unit)

        # Backward selection
        if not self.backprop_selection_frozen:
            # If backprop_selection is not frozen, backprop layer/unit follows selected unit
            if not (self.backprop_layer == self.layer and self.backprop_unit == self.selected_unit):
                self.backprop_layer = self.layer
                self.backprop_unit = self.selected_unit
                self.back_stale = True    # If there is any change, back diffs are now stale
