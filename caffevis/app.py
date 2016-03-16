#! /usr/bin/env python
# -*- coding: utf-8

import sys
import os
import cv2
import numpy as np
import time
import StringIO

from misc import WithTimer
from numpy_cache import FIFOLimitedArrayCache
from app_base import BaseApp
from image_misc import norm01, norm01c, norm0255, tile_images_normalize, ensure_float01, tile_images_make_tiles, ensure_uint255_and_resize_to_fit, get_tiles_height_width, get_tiles_height_width_ratio
from image_misc import FormattedString, cv2_typeset_text, to_255
from caffe_proc_thread import CaffeProcThread
from jpg_vis_loading_thread import JPGVisLoadingThread
from caffevis_app_state import CaffeVisAppState
from caffevis_helper import get_pretty_layer_name, read_label_file, load_sprite_image, load_square_sprite_image, check_force_backward_true



class CaffeVisApp(BaseApp):
    '''App to visualize using caffe.'''

    def __init__(self, settings, key_bindings):
        super(CaffeVisApp, self).__init__(settings, key_bindings)
        print 'Got settings', settings
        self.settings = settings
        self.bindings = key_bindings

        self._net_channel_swap = (2,1,0)
        self._net_channel_swap_inv = tuple([self._net_channel_swap.index(ii) for ii in range(len(self._net_channel_swap))])
        self._range_scale = 1.0      # not needed; image already in [0,255]

        # Set the mode to CPU or GPU. Note: in the latest Caffe
        # versions, there is one Caffe object *per thread*, so the
        # mode must be set per thread! Here we set the mode for the
        # main thread; it is also separately set in CaffeProcThread.
        sys.path.insert(0, os.path.join(settings.caffevis_caffe_root, 'python'))
        import caffe
        if settings.caffevis_mode_gpu:
            caffe.set_mode_gpu()
            print 'CaffeVisApp mode (in main thread):     GPU'
        else:
            caffe.set_mode_cpu()
            print 'CaffeVisApp mode (in main thread):     CPU'
        self.net = caffe.Classifier(
            settings.caffevis_deploy_prototxt,
            settings.caffevis_network_weights,
            mean = None,                                 # Set to None for now, assign later         # self._data_mean,
            channel_swap = self._net_channel_swap,
            raw_scale = self._range_scale,
        )

        if isinstance(settings.caffevis_data_mean, basestring):
            # If the mean is given as a filename, load the file
            try:
                self._data_mean = np.load(settings.caffevis_data_mean)
            except IOError:
                print '\n\nCound not load mean file:', settings.caffevis_data_mean
                print 'Ensure that the values in settings.py point to a valid model weights file, network'
                print 'definition prototxt, and mean. To fetch a default model and mean file, use:\n'
                print '$ cd models/caffenet-yos/'
                print '$ ./fetch.sh\n\n'
                raise
            input_shape = self.net.blobs[self.net.inputs[0]].data.shape[-2:]   # e.g. 227x227
            # Crop center region (e.g. 227x227) if mean is larger (e.g. 256x256)
            excess_h = self._data_mean.shape[1] - input_shape[0]
            excess_w = self._data_mean.shape[2] - input_shape[1]
            assert excess_h >= 0 and excess_w >= 0, 'mean should be at least as large as %s' % repr(input_shape)
            self._data_mean = self._data_mean[:, (excess_h/2):(excess_h/2+input_shape[0]),
                                              (excess_w/2):(excess_w/2+input_shape[1])]
        elif settings.caffevis_data_mean is None:
            self._data_mean = None
        else:
            # The mean has been given as a value or a tuple of values
            self._data_mean = np.array(settings.caffevis_data_mean)
            # Promote to shape C,1,1
            while len(self._data_mean.shape) < 1:
                self._data_mean = np.expand_dims(self._data_mean, -1)
            
            #if not isinstance(self._data_mean, tuple):
            #    # If given as int/float: promote to tuple
            #    self._data_mean = tuple(self._data_mean)
        if self._data_mean is not None:
            self.net.transformer.set_mean(self.net.inputs[0], self._data_mean)
        
        check_force_backward_true(settings.caffevis_deploy_prototxt)

        self.labels = None
        if self.settings.caffevis_labels:
            self.labels = read_label_file(self.settings.caffevis_labels)
        self.proc_thread = None
        self.jpgvis_thread = None
        self.handled_frames = 0
        if settings.caffevis_jpg_cache_size < 10*1024**2:
            raise Exception('caffevis_jpg_cache_size must be at least 10MB for normal operation.')
        self.img_cache = FIFOLimitedArrayCache(settings.caffevis_jpg_cache_size)

        self._populate_net_layer_info()

    def _populate_net_layer_info(self):
        '''For each layer, save the number of filters and precompute
        tile arrangement (needed by CaffeVisAppState to handle
        keyboard navigation).
        '''
        self.net_layer_info = {}
        for key in self.net.blobs.keys():
            self.net_layer_info[key] = {}
            # Conv example: (1, 96, 55, 55)
            # FC example: (1, 1000)
            blob_shape = self.net.blobs[key].data.shape
            assert len(blob_shape) in (2,4), 'Expected either 2 for FC or 4 for conv layer'
            self.net_layer_info[key]['isconv'] = (len(blob_shape) == 4)
            self.net_layer_info[key]['data_shape'] = blob_shape[1:]  # Chop off batch size
            self.net_layer_info[key]['n_tiles'] = blob_shape[1]
            self.net_layer_info[key]['tiles_rc'] = get_tiles_height_width_ratio(blob_shape[1], self.settings.caffevis_layers_aspect_ratio)
            self.net_layer_info[key]['tile_rows'] = self.net_layer_info[key]['tiles_rc'][0]
            self.net_layer_info[key]['tile_cols'] = self.net_layer_info[key]['tiles_rc'][1]

    def start(self):
        self.state = CaffeVisAppState(self.net, self.settings, self.bindings, self.net_layer_info)
        self.state.drawing_stale = True
        self.layer_print_names = [get_pretty_layer_name(self.settings, nn) for nn in self.state._layers]

        if self.proc_thread is None or not self.proc_thread.is_alive():
            # Start thread if it's not already running
            self.proc_thread = CaffeProcThread(self.net, self.state,
                                               self.settings.caffevis_frame_wait_sleep,
                                               self.settings.caffevis_pause_after_keys,
                                               self.settings.caffevis_heartbeat_required,
                                               self.settings.caffevis_mode_gpu)
            self.proc_thread.start()

        if self.jpgvis_thread is None or not self.jpgvis_thread.is_alive():
            # Start thread if it's not already running
            self.jpgvis_thread = JPGVisLoadingThread(self.settings, self.state, self.img_cache,
                                                     self.settings.caffevis_jpg_load_sleep,
                                                     self.settings.caffevis_heartbeat_required)
            self.jpgvis_thread.start()
                

    def get_heartbeats(self):
        return [self.proc_thread.heartbeat, self.jpgvis_thread.heartbeat]
            
    def quit(self):
        print 'CaffeVisApp: trying to quit'

        with self.state.lock:
            self.state.quit = True

        if self.proc_thread != None:
            for ii in range(3):
                self.proc_thread.join(1)
                if not self.proc_thread.is_alive():
                    break
            if self.proc_thread.is_alive():
                raise Exception('CaffeVisApp: Could not join proc_thread; giving up.')
            self.proc_thread = None
                
        print 'CaffeVisApp: quitting.'
        
    def _can_skip_all(self, panes):
        return ('caffevis_layers' not in panes.keys())
        
    def handle_input(self, input_image, panes):
        if self.debug_level > 1:
            print 'handle_input: frame number', self.handled_frames, 'is', 'None' if input_image is None else 'Available'
        self.handled_frames += 1
        if self._can_skip_all(panes):
            return

        with self.state.lock:
            if self.debug_level > 1:
                print 'CaffeVisApp.handle_input: pushed frame'
            self.state.next_frame = input_image
            if self.debug_level > 1:
                print 'CaffeVisApp.handle_input: caffe_net_state is:', self.state.caffe_net_state
    
    def redraw_needed(self):
        return self.state.redraw_needed()

    def draw(self, panes):
        if self._can_skip_all(panes):
            if self.debug_level > 1:
                print 'CaffeVisApp.draw: skipping'
            return False

        with self.state.lock:
            # Hold lock throughout drawing
            do_draw = self.state.drawing_stale and self.state.caffe_net_state == 'free'
            #print 'CaffeProcThread.draw: caffe_net_state is:', self.state.caffe_net_state
            if do_draw:
                self.state.caffe_net_state = 'draw'

        if do_draw:
            if self.debug_level > 1:
                print 'CaffeVisApp.draw: drawing'

            if 'caffevis_control' in panes:
                self._draw_control_pane(panes['caffevis_control'])
            if 'caffevis_status' in panes:
                self._draw_status_pane(panes['caffevis_status'])
            layer_data_3D_highres = None
            if 'caffevis_layers' in panes:
                layer_data_3D_highres = self._draw_layer_pane(panes['caffevis_layers'])
            if 'caffevis_aux' in panes:
                self._draw_aux_pane(panes['caffevis_aux'], layer_data_3D_highres)
            if 'caffevis_back' in panes:
                # Draw back pane as normal
                self._draw_back_pane(panes['caffevis_back'])
                if self.state.layers_pane_zoom_mode == 2:
                    # ALSO draw back pane into layers pane
                    self._draw_back_pane(panes['caffevis_layers'])
            if 'caffevis_jpgvis' in panes:
                self._draw_jpgvis_pane(panes['caffevis_jpgvis'])

            with self.state.lock:
                self.state.drawing_stale = False
                self.state.caffe_net_state = 'free'
        return do_draw

    def _draw_prob_labels_pane(self, pane):
        '''Adds text label annotation atop the given pane.'''

        if not self.labels or not self.state.show_label_predictions or not self.settings.caffevis_prob_layer:
            return

        #pane.data[:] = to_255(self.settings.window_background)
        defaults = {'face':  getattr(cv2, self.settings.caffevis_class_face),
                    'fsize': self.settings.caffevis_class_fsize,
                    'clr':   to_255(self.settings.caffevis_class_clr_0),
                    'thick': self.settings.caffevis_class_thick}
        loc = self.settings.caffevis_class_loc[::-1]   # Reverse to OpenCV c,r order
        clr_0 = to_255(self.settings.caffevis_class_clr_0)
        clr_1 = to_255(self.settings.caffevis_class_clr_1)

        probs_flat = self.net.blobs[self.settings.caffevis_prob_layer].data.flatten()
        top_5 = probs_flat.argsort()[-1:-6:-1]

        strings = []
        pmax = probs_flat[top_5[0]]
        for idx in top_5:
            prob = probs_flat[idx]
            text = '%.2f %s' % (prob, self.labels[idx])
            fs = FormattedString(text, defaults)
            #fs.clr = tuple([clr_1[ii]*prob/pmax + clr_0[ii]*(1-prob/pmax) for ii in range(3)])
            fs.clr = tuple([max(0,min(255,clr_1[ii]*prob + clr_0[ii]*(1-prob))) for ii in range(3)])
            strings.append([fs])   # Line contains just fs

        cv2_typeset_text(pane.data, strings, loc,
                         line_spacing = self.settings.caffevis_class_line_spacing)

    def _draw_control_pane(self, pane):
        pane.data[:] = to_255(self.settings.window_background)

        with self.state.lock:
            layer_idx = self.state.layer_idx

        loc = self.settings.caffevis_control_loc[::-1]   # Reverse to OpenCV c,r order

        strings = []
        defaults = {'face':  getattr(cv2, self.settings.caffevis_control_face),
                    'fsize': self.settings.caffevis_control_fsize,
                    'clr':   to_255(self.settings.caffevis_control_clr),
                    'thick': self.settings.caffevis_control_thick}

        for ii in range(len(self.layer_print_names)):
            fs = FormattedString(self.layer_print_names[ii], defaults)
            this_layer = self.state._layers[ii]
            if self.state.backprop_selection_frozen and this_layer == self.state.backprop_layer:
                fs.clr   = to_255(self.settings.caffevis_control_clr_bp)
                fs.thick = self.settings.caffevis_control_thick_bp
            if this_layer == self.state.layer:
                if self.state.cursor_area == 'top':
                    fs.clr = to_255(self.settings.caffevis_control_clr_cursor)
                    fs.thick = self.settings.caffevis_control_thick_cursor
                else:
                    if not (self.state.backprop_selection_frozen and this_layer == self.state.backprop_layer):
                        fs.clr = to_255(self.settings.caffevis_control_clr_selected)
                        fs.thick = self.settings.caffevis_control_thick_selected
            strings.append(fs)

        cv2_typeset_text(pane.data, strings, loc,
                         line_spacing = self.settings.caffevis_control_line_spacing,
                         wrap = True)

    def _draw_status_pane(self, pane):
        pane.data[:] = to_255(self.settings.window_background)

        defaults = {'face':  getattr(cv2, self.settings.caffevis_status_face),
                    'fsize': self.settings.caffevis_status_fsize,
                    'clr':   to_255(self.settings.caffevis_status_clr),
                    'thick': self.settings.caffevis_status_thick}
        loc = self.settings.caffevis_status_loc[::-1]   # Reverse to OpenCV c,r order

        status = StringIO.StringIO()
        fps = self.proc_thread.approx_fps()
        with self.state.lock:
            print >>status, 'pattern' if self.state.pattern_mode else ('back' if self.state.layers_show_back else 'fwd'),
            print >>status, '%s:%d |' % (self.state.layer, self.state.selected_unit),
            if not self.state.back_enabled:
                print >>status, 'Back: off',
            else:
                print >>status, 'Back: %s' % ('deconv' if self.state.back_mode == 'deconv' else 'bprop'),
                print >>status, '(from %s_%d, disp %s)' % (self.state.backprop_layer,
                                                           self.state.backprop_unit,
                                                           self.state.back_filt_mode),
            print >>status, '|',
            print >>status, 'Boost: %g/%g' % (self.state.layer_boost_indiv, self.state.layer_boost_gamma)

            if fps > 0:
                print >>status, '| FPS: %.01f' % fps

            if self.state.extra_msg:
                print >>status, '|', self.state.extra_msg
                self.state.extra_msg = ''

        strings = [FormattedString(line, defaults) for line in status.getvalue().split('\n')]

        cv2_typeset_text(pane.data, strings, loc,
                         line_spacing = self.settings.caffevis_status_line_spacing)
    
    def _draw_layer_pane(self, pane):
        '''Returns the data shown in highres format, b01c order.'''
        
        if self.state.layers_show_back:
            layer_dat_3D = self.net.blobs[self.state.layer].diff[0]
        else:
            layer_dat_3D = self.net.blobs[self.state.layer].data[0]
        # Promote FC layers with shape (n) to have shape (n,1,1)
        if len(layer_dat_3D.shape) == 1:
            layer_dat_3D = layer_dat_3D[:,np.newaxis,np.newaxis]

        n_tiles = layer_dat_3D.shape[0]
        tile_rows,tile_cols = self.net_layer_info[self.state.layer]['tiles_rc']

        display_3D_highres = None
        if self.state.pattern_mode:
            # Show desired patterns loaded from disk

            load_layer = self.state.layer
            if self.settings.caffevis_jpgvis_remap and self.state.layer in self.settings.caffevis_jpgvis_remap:
                load_layer = self.settings.caffevis_jpgvis_remap[self.state.layer]

            
            if self.settings.caffevis_jpgvis_layers and load_layer in self.settings.caffevis_jpgvis_layers:
                jpg_path = os.path.join(self.settings.caffevis_unit_jpg_dir,
                                        'regularized_opt', load_layer, 'whole_layer.jpg')

                # Get highres version
                #cache_before = str(self.img_cache)
                display_3D_highres = self.img_cache.get((jpg_path, 'whole'), None)
                #else:
                #    display_3D_highres = None

                if display_3D_highres is None:
                    try:
                        with WithTimer('CaffeVisApp:load_sprite_image', quiet = self.debug_level < 1):
                            display_3D_highres = load_square_sprite_image(jpg_path, n_sprites = n_tiles)
                    except IOError:
                        # File does not exist, so just display disabled.
                        pass
                    else:
                        self.img_cache.set((jpg_path, 'whole'), display_3D_highres)
                #cache_after = str(self.img_cache)
                #print 'Cache was / is:\n  %s\n  %s' % (cache_before, cache_after)

            if display_3D_highres is not None:
                # Get lowres version, maybe. Assume we want at least one pixel for selection border.
                row_downsamp_factor = int(np.ceil(float(display_3D_highres.shape[1]) / (pane.data.shape[0] / tile_rows - 2)))
                col_downsamp_factor = int(np.ceil(float(display_3D_highres.shape[2]) / (pane.data.shape[1] / tile_cols - 2)))
                ds = max(row_downsamp_factor, col_downsamp_factor)
                if ds > 1:
                    #print 'Downsampling by', ds
                    display_3D = display_3D_highres[:,::ds,::ds,:]
                else:
                    display_3D = display_3D_highres
            else:
                display_3D = layer_dat_3D * 0  # nothing to show

        else:

            # Show data from network (activations or diffs)
            if self.state.layers_show_back:
                back_what_to_disp = self.get_back_what_to_disp()
                if back_what_to_disp == 'disabled':
                    layer_dat_3D_normalized = np.tile(self.settings.window_background, layer_dat_3D.shape + (1,))
                elif back_what_to_disp == 'stale':
                    layer_dat_3D_normalized = np.tile(self.settings.stale_background, layer_dat_3D.shape + (1,))
                else:
                    layer_dat_3D_normalized = tile_images_normalize(layer_dat_3D,
                                                                    boost_indiv = self.state.layer_boost_indiv,
                                                                    boost_gamma = self.state.layer_boost_gamma,
                                                                    neg_pos_colors = ((1,0,0), (0,1,0)))
            else:
                layer_dat_3D_normalized = tile_images_normalize(layer_dat_3D,
                                                                boost_indiv = self.state.layer_boost_indiv,
                                                                boost_gamma = self.state.layer_boost_gamma)
            #print ' ===layer_dat_3D_normalized.shape', layer_dat_3D_normalized.shape, 'layer_dat_3D_normalized dtype', layer_dat_3D_normalized.dtype, 'range', layer_dat_3D_normalized.min(), layer_dat_3D_normalized.max()

            display_3D         = layer_dat_3D_normalized

        # Convert to float if necessary:
        display_3D = ensure_float01(display_3D)
        # Upsample gray -> color if necessary
        #   e.g. (1000,32,32) -> (1000,32,32,3)
        if len(display_3D.shape) == 3:
            display_3D = display_3D[:,:,:,np.newaxis]
        if display_3D.shape[3] == 1:
            display_3D = np.tile(display_3D, (1, 1, 1, 3))
        # Upsample unit length tiles to give a more sane tile / highlight ratio
        #   e.g. (1000,1,1,3) -> (1000,3,3,3)
        if display_3D.shape[1] == 1:
            display_3D = np.tile(display_3D, (1, 3, 3, 1))
        if self.state.layers_show_back and not self.state.pattern_mode:
            padval = self.settings.caffevis_layer_clr_back_background
        else:
            padval = self.settings.window_background

        highlights = [None] * n_tiles
        with self.state.lock:
            if self.state.cursor_area == 'bottom':
                highlights[self.state.selected_unit] = self.settings.caffevis_layer_clr_cursor  # in [0,1] range
            if self.state.backprop_selection_frozen and self.state.layer == self.state.backprop_layer:
                highlights[self.state.backprop_unit] = self.settings.caffevis_layer_clr_back_sel  # in [0,1] range

        _, display_2D = tile_images_make_tiles(display_3D, hw = (tile_rows,tile_cols), padval = padval, highlights = highlights)

        if display_3D_highres is None:
            display_3D_highres = display_3D
        
        # Display pane based on layers_pane_zoom_mode
        state_layers_pane_zoom_mode = self.state.layers_pane_zoom_mode
        assert state_layers_pane_zoom_mode in (0,1,2)
        if state_layers_pane_zoom_mode == 0:
            # Mode 0: normal display (activations or patterns)
            display_2D_resize = ensure_uint255_and_resize_to_fit(display_2D, pane.data.shape)
        elif state_layers_pane_zoom_mode == 1:
            # Mode 1: zoomed selection
            unit_data = display_3D_highres[self.state.selected_unit]
            display_2D_resize = ensure_uint255_and_resize_to_fit(unit_data, pane.data.shape)
        else:
            # Mode 2: zoomed backprop pane
            display_2D_resize = ensure_uint255_and_resize_to_fit(display_2D, pane.data.shape) * 0

        pane.data[:] = to_255(self.settings.window_background)
        pane.data[0:display_2D_resize.shape[0], 0:display_2D_resize.shape[1], :] = display_2D_resize
        
        if self.settings.caffevis_label_layers and self.state.layer in self.settings.caffevis_label_layers and self.labels and self.state.cursor_area == 'bottom':
            # Display label annotation atop layers pane (e.g. for fc8/prob)
            defaults = {'face':  getattr(cv2, self.settings.caffevis_label_face),
                        'fsize': self.settings.caffevis_label_fsize,
                        'clr':   to_255(self.settings.caffevis_label_clr),
                        'thick': self.settings.caffevis_label_thick}
            loc_base = self.settings.caffevis_label_loc[::-1]   # Reverse to OpenCV c,r order
            lines = [FormattedString(self.labels[self.state.selected_unit], defaults)]
            cv2_typeset_text(pane.data, lines, loc_base)
            
        return display_3D_highres

    def _draw_aux_pane(self, pane, layer_data_normalized):
        pane.data[:] = to_255(self.settings.window_background)

        mode = None
        with self.state.lock:
            if self.state.cursor_area == 'bottom':
                mode = 'selected'
            else:
                mode = 'prob_labels'
                
        if mode == 'selected':
            unit_data = layer_data_normalized[self.state.selected_unit]
            unit_data_resize = ensure_uint255_and_resize_to_fit(unit_data, pane.data.shape)
            pane.data[0:unit_data_resize.shape[0], 0:unit_data_resize.shape[1], :] = unit_data_resize
        elif mode == 'prob_labels':
            self._draw_prob_labels_pane(pane)

    def _draw_back_pane(self, pane):
        mode = None
        with self.state.lock:
            back_enabled = self.state.back_enabled
            back_mode = self.state.back_mode
            back_filt_mode = self.state.back_filt_mode
            state_layer = self.state.layer
            selected_unit = self.state.selected_unit
            back_what_to_disp = self.get_back_what_to_disp()
                
        if back_what_to_disp == 'disabled':
            pane.data[:] = to_255(self.settings.window_background)

        elif back_what_to_disp == 'stale':
            pane.data[:] = to_255(self.settings.stale_background)

        else:
            # One of the backprop modes is enabled and the back computation (gradient or deconv) is up to date
            
            grad_blob = self.net.blobs['data'].diff

            # Manually deprocess (skip mean subtraction and rescaling)
            #grad_img = self.net.deprocess('data', diff_blob)
            grad_blob = grad_blob[0]                    # bc01 -> c01
            grad_blob = grad_blob.transpose((1,2,0))    # c01 -> 01c
            grad_img = grad_blob[:, :, self._net_channel_swap_inv]  # e.g. BGR -> RGB

            # Mode-specific processing
            assert back_mode in ('grad', 'deconv')
            assert back_filt_mode in ('raw', 'gray', 'norm', 'normblur')
            if back_filt_mode == 'raw':
                grad_img = norm01c(grad_img, 0)
            elif back_filt_mode == 'gray':
                grad_img = grad_img.mean(axis=2)
                grad_img = norm01c(grad_img, 0)
            elif back_filt_mode == 'norm':
                grad_img = np.linalg.norm(grad_img, axis=2)
                grad_img = norm01(grad_img)
            else:
                grad_img = np.linalg.norm(grad_img, axis=2)
                cv2.GaussianBlur(grad_img, (0,0), self.settings.caffevis_grad_norm_blur_radius, grad_img)
                grad_img = norm01(grad_img)

            # If necessary, re-promote from grayscale to color
            if len(grad_img.shape) == 2:
                grad_img = np.tile(grad_img[:,:,np.newaxis], 3)

            grad_img_resize = ensure_uint255_and_resize_to_fit(grad_img, pane.data.shape)

            pane.data[0:grad_img_resize.shape[0], 0:grad_img_resize.shape[1], :] = grad_img_resize

    def _draw_jpgvis_pane(self, pane):
        pane.data[:] = to_255(self.settings.window_background)

        with self.state.lock:
            state_layer, state_selected_unit, cursor_area, show_unit_jpgs = self.state.layer, self.state.selected_unit, self.state.cursor_area, self.state.show_unit_jpgs

        try:
            # Some may be missing this setting
            self.settings.caffevis_jpgvis_layers
        except:
            print '\n\nNOTE: you need to upgrade your settings.py and settings_local.py files. See README.md.\n\n'
            raise
            
        if self.settings.caffevis_jpgvis_remap and state_layer in self.settings.caffevis_jpgvis_remap:
            img_key_layer = self.settings.caffevis_jpgvis_remap[state_layer]
        else:
            img_key_layer = state_layer

        if self.settings.caffevis_jpgvis_layers and img_key_layer in self.settings.caffevis_jpgvis_layers and cursor_area == 'bottom' and show_unit_jpgs:
            img_key = (img_key_layer, state_selected_unit, pane.data.shape)
            img_resize = self.img_cache.get(img_key, None)
            if img_resize is None:
                # If img_resize is None, loading has not yet been attempted, so show stale image and request load by JPGVisLoadingThread
                with self.state.lock:
                    self.state.jpgvis_to_load_key = img_key
                pane.data[:] = to_255(self.settings.stale_background)
            elif img_resize.nbytes == 0:
                # This is the sentinal value when the image is not
                # found, i.e. loading was already attempted but no jpg
                # assets were found. Just display disabled.
                pane.data[:] = to_255(self.settings.window_background)
            else:
                # Show image
                pane.data[:img_resize.shape[0], :img_resize.shape[1], :] = img_resize
        else:
            # Will never be available
            pane.data[:] = to_255(self.settings.window_background)

    def handle_key(self, key, panes):
        return self.state.handle_key(key)

    def get_back_what_to_disp(self):
        '''Whether to show back diff information or stale or disabled indicator'''
        if (self.state.cursor_area == 'top' and not self.state.backprop_selection_frozen) or not self.state.back_enabled:
            return 'disabled'
        elif self.state.back_stale:
            return 'stale'
        else:
            return 'normal'

    def set_debug(self, level):
        self.debug_level = level
        self.proc_thread.debug_level = level
        self.jpgvis_thread.debug_level = level

    def draw_help(self, help_pane, locy):
        defaults = {'face':  getattr(cv2, self.settings.help_face),
                    'fsize': self.settings.help_fsize,
                    'clr':   to_255(self.settings.help_clr),
                    'thick': self.settings.help_thick}
        loc_base = self.settings.help_loc[::-1]   # Reverse to OpenCV c,r order
        locx = loc_base[0]

        lines = []
        lines.append([FormattedString('', defaults)])
        lines.append([FormattedString('Caffevis keys', defaults)])
        
        kl,_ = self.bindings.get_key_help('sel_left')
        kr,_ = self.bindings.get_key_help('sel_right')
        ku,_ = self.bindings.get_key_help('sel_up')
        kd,_ = self.bindings.get_key_help('sel_down')
        klf,_ = self.bindings.get_key_help('sel_left_fast')
        krf,_ = self.bindings.get_key_help('sel_right_fast')
        kuf,_ = self.bindings.get_key_help('sel_up_fast')
        kdf,_ = self.bindings.get_key_help('sel_down_fast')

        keys_nav_0 = ','.join([kk[0] for kk in (kl, kr, ku, kd)])
        keys_nav_1 = ''
        if len(kl)>1 and len(kr)>1 and len(ku)>1 and len(kd)>1:
            keys_nav_1 += ' or '
            keys_nav_1 += ','.join([kk[1] for kk in (kl, kr, ku, kd)])
        keys_nav_f = ','.join([kk[0] for kk in (klf, krf, kuf, kdf)])
        nav_string = 'Navigate with %s%s. Use %s to move faster.' % (keys_nav_0, keys_nav_1, keys_nav_f)
        lines.append([FormattedString('', defaults, width=120, align='right'),
                      FormattedString(nav_string, defaults)])
            
        for tag in ('sel_layer_left', 'sel_layer_right', 'zoom_mode', 'pattern_mode',
                    'ez_back_mode_loop', 'freeze_back_unit', 'show_back', 'back_mode', 'back_filt_mode',
                    'boost_gamma', 'boost_individual', 'reset_state'):
            key_strings, help_string = self.bindings.get_key_help(tag)
            label = '%10s:' % (','.join(key_strings))
            lines.append([FormattedString(label, defaults, width=120, align='right'),
                          FormattedString(help_string, defaults)])

        locy = cv2_typeset_text(help_pane.data, lines, (locx, locy),
                                line_spacing = self.settings.help_line_spacing)

        return locy
