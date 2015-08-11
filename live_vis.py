#! /usr/bin/env python

import os
import sys
import importlib
from collections import OrderedDict
import numpy as np
from threading import Lock, RLock, Thread
import time
import glob
import re

try:
    import cv2
except ImportError:
    print 'Error: Could not import cv2, please install it first.'
    raise

from misc import WithTimer
from image_misc import cv2_imshow_rgb, cv2_read_file_rgb, read_cam_frame, crop_to_square
from image_misc import FormattedString, cv2_typeset_text, to_255
from bindings import bindings



class ImproperlyConfigured(Exception):
    pass



class Pane(object):
    '''Hold info about one window pane (rectangular region within the main window)'''

    def __init__(self, i_begin, j_begin, i_size, j_size):
        self.i_begin = i_begin
        self.j_begin = j_begin
        self.i_size = i_size
        self.j_size = j_size
        self.i_end = i_begin + i_size
        self.j_end = j_begin + j_size
        self.data = None    # eventually contains a slice of the window buffer



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



class InputImageFetcher(CodependentThread):
    '''Fetches images from a webcam or loads from a directory.'''
    
    def __init__(self, settings):
        CodependentThread.__init__(self, settings.input_updater_heartbeat_required)
        self.daemon = True
        self.lock = RLock()
        self.quit = False
        self.latest_frame_idx = -1
        self.latest_frame_data = None
        self.latest_frame_is_from_cam = False
        self.static_file_mode = True
        self.settings = settings
        self.static_file_stretch_mode = self.settings.static_file_stretch_mode
        
        # Cam input
        self.capture_device = settings.input_updater_capture_device
        self.no_cam_present = (self.capture_device is None)     # Disable all cam functionality
        self.bound_cap_device = None
        self.sleep_after_read_frame = settings.input_updater_sleep_after_read_frame
        self.latest_cam_frame = None
        self.freeze_cam = False

        # Static file input
        self.latest_static_filename = None
        self.latest_static_frame = None
        self.static_file_idx = None
        self.static_file_idx_increment = 0
        
    def bind_camera(self):
        # Due to OpenCV limitations, this should be called from the main thread
        print 'InputImageFetcher: bind_camera starting'
        if self.no_cam_present:
            print 'InputImageFetcher: skipping camera bind (device: None)'
        else:
            self.bound_cap_device = cv2.VideoCapture(self.capture_device)
            if self.bound_cap_device.isOpened():
                print 'InputImageFetcher: capture device %s is open' % self.capture_device
            else:
                print 'InputImageFetcher: capture device %s failed to open! Camera will not be available!\n\n' % self.capture_device
        print 'InputImageFetcher: bind_camera finished'

    def free_camera(self):
        # Due to OpenCV limitations, this should be called from the main thread
        if self.no_cam_present:
            print 'InputImageFetcher: skipping camera free (device: None)'
        else:
            print 'InputImageFetcher: freeing camera'
            del self.bound_cap_device  # free the camera
            self.bound_cap_device = None
            print 'InputImageFetcher: camera freed'

    def set_mode_static(self):
        with self.lock:
            self.static_file_mode = True
        
    def set_mode_cam(self):
        with self.lock:
            if self.no_cam_present:
                print 'WARNING: ignoring set_mode_cam, no cam present'
            else:
                self.static_file_mode = False
                assert self.bound_cap_device != None, 'Call bind_camera first'
        
    def toggle_input_mode(self):
        with self.lock:
            if self.static_file_mode:
                self.set_mode_cam()
            else:
                self.set_mode_static()
        
    def set_mode_stretch_on(self):
        with self.lock:
            if not self.static_file_stretch_mode:
                self.static_file_stretch_mode = True
                self.latest_static_frame = None   # Force reload
                #self.latest_frame_is_from_cam = True  # Force reload
        
    def set_mode_stretch_off(self):
        with self.lock:
            if self.static_file_stretch_mode:
                self.static_file_stretch_mode = False
                self.latest_static_frame = None   # Force reload
                #self.latest_frame_is_from_cam = True  # Force reload
        
    def toggle_stretch_mode(self):
        with self.lock:
            if self.static_file_stretch_mode:
                self.set_mode_stretch_off()
            else:
                self.set_mode_stretch_on()
        
    def run(self):
        while not self.quit and not self.is_timed_out():
            #start_time = time.time()
            if self.static_file_mode:
                self.check_increment_and_load_image()
            else:
                if self.freeze_cam and self.latest_cam_frame is not None:
                    # If static file mode was switched to cam mode but cam is still frozen, we need to push the cam frame again
                    if not self.latest_frame_is_from_cam:
                        self._increment_and_set_frame(self.latest_cam_frame, True)
                else:
                    frame_full = read_cam_frame(self.bound_cap_device)
                    #print '====> just read frame', frame_full.shape
                    frame = crop_to_square(frame_full)
                    with self.lock:
                        self.latest_cam_frame = frame
                        self._increment_and_set_frame(self.latest_cam_frame, True)
            
            time.sleep(self.sleep_after_read_frame)
            #print 'Reading one frame took', time.time() - start_time

        print 'InputImageFetcher: exiting run method'
        #print 'InputImageFetcher: read', self.read_frames, 'frames'

    def get_frame(self):
        '''Fetch the latest frame_idx and frame. The idx increments
        any time the frame data changes. If the idx is < 0, the frame
        is not valid.
        '''
        with self.lock:
            return (self.latest_frame_idx, self.latest_frame_data)

    def increment_static_file_idx(self, amount = 1):
        with self.lock:
            self.static_file_idx_increment += amount

    def _increment_and_set_frame(self, frame, from_cam):
        assert frame is not None
        with self.lock:
            self.latest_frame_idx += 1
            self.latest_frame_data = frame
            self.latest_frame_is_from_cam = from_cam

    def check_increment_and_load_image(self):
        with self.lock:
            if (self.static_file_idx_increment == 0 and
                self.static_file_idx is not None and
                not self.latest_frame_is_from_cam and
                self.latest_static_frame is not None):
                # Skip if a static frame is already loaded and there is no increment
                return
            available_files = []
            match_flags = re.IGNORECASE if self.settings.static_files_ignore_case else 0
            for filename in os.listdir(self.settings.static_files_dir):
                if re.match(self.settings.static_files_regexp, filename, match_flags):
                    available_files.append(filename)
            #print 'Found files:'
            #for filename in available_files:
            #    print '   %s' % filename
            assert len(available_files) != 0, ('Error: No files found in %s matching %s (current working directory is %s)' %
                                               (self.settings.static_files_dir, self.settings.static_files_regexp, os.getcwd()))
            if self.static_file_idx is None:
                self.static_file_idx = 0
            self.static_file_idx = (self.static_file_idx + self.static_file_idx_increment) % len(available_files)
            self.static_file_idx_increment = 0
            if self.latest_static_filename != available_files[self.static_file_idx] or self.latest_static_frame is None:
                self.latest_static_filename = available_files[self.static_file_idx]
                im = cv2_read_file_rgb(os.path.join(self.settings.static_files_dir, self.latest_static_filename))
                if not self.static_file_stretch_mode:
                    im = crop_to_square(im)
                self.latest_static_frame = im
            self._increment_and_set_frame(self.latest_static_frame, False)



class LiveVis(object):
    '''Runs the demo'''

    def __init__(self, settings):
        self.settings = settings
        self.bindings = bindings
        
        self.app_classes = OrderedDict()
        self.apps = OrderedDict()
        
        for module_path, app_name in settings.installed_apps:
            module = importlib.import_module(module_path)
            print 'got module', module
            app_class  = getattr(module, app_name)
            print 'got app', app_class
            self.app_classes[app_name] = app_class

        for app_name, app_class in self.app_classes.iteritems():
            app = app_class(settings, self.bindings)
            self.apps[app_name] = app
        self.help_mode = False
        self.window_name = 'Deep Visualization Toolbox'    
        self.quit = False
        self.debug_level = 0

    def init_window(self):
        cv2.namedWindow(self.window_name)
        max_i, max_j = 0, 0
        if len(self.settings.window_panes) == 0:
            raise ImproperlyConfigured('settings.window_panes is empty.')
        self.panes = OrderedDict()
        for pane_name, pane_dimensions in self.settings.window_panes:
            if len(pane_dimensions) != 4:
                raise ImproperlyConfigured('pane dimensions should be a tuple of length 4, but it is "%s"' % repr(pane_dimensions))
            i_begin, j_begin, i_size, j_size = pane_dimensions
            max_i = max(max_i, i_begin + i_size)
            max_j = max(max_j, j_begin + j_size)
            self.panes[pane_name] = Pane(i_begin, j_begin, i_size, j_size)
        self.buffer_height = max_i
        self.buffer_width = max_j

        self.window_buffer = np.tile(np.array(np.array(self.settings.window_background) * 255, 'uint8'),
                                     (max_i,max_j,1))
        #print 'BUFFER IS:', self.window_buffer.shape, self.window_buffer.min(), self.window_buffer.max()

        for _,pane in self.panes.iteritems():
            pane.data = self.window_buffer[pane.i_begin:pane.i_end, pane.j_begin:pane.j_end]

        # Allocate help pane
        for ll in self.settings.help_pane_loc:
            assert ll >= 0 and ll <= 1, 'help_pane_loc values should be in [0,1]'
        self.help_pane = Pane(int(self.settings.help_pane_loc[0]*max_i),
                              int(self.settings.help_pane_loc[1]*max_j),
                              int(self.settings.help_pane_loc[2]*max_i),
                              int(self.settings.help_pane_loc[3]*max_j))
        self.help_buffer = self.window_buffer.copy() # For rendering help mode
        self.help_pane.data = self.help_buffer[self.help_pane.i_begin:self.help_pane.i_end, self.help_pane.j_begin:self.help_pane.j_end]

    def run_loop(self):
        self.quit = False
        # Setup
        self.init_window()
        #cap = cv2.VideoCapture(self.settings.capture_device)
        self.input_updater = InputImageFetcher(self.settings)
        self.input_updater.bind_camera()
        self.input_updater.start()

        heartbeat_functions = [self.input_updater.heartbeat]
        for app_name, app in self.apps.iteritems():
            print 'Starting app:', app_name
            app.start()
            heartbeat_functions.extend(app.get_heartbeats())

        ii = 0
        since_keypress = 999
        since_redraw = 999
        since_imshow = 0
        last_render = time.time() - 999
        latest_frame_idx = None
        latest_frame_data = None
        frame_for_apps = None
        redraw_needed = True    # Force redraw the first time
        imshow_needed = True
        while not self.quit:
            # Call any heartbeats
            for heartbeat in heartbeat_functions:
                #print 'Heartbeat: calling', heartbeat
                heartbeat()
            
            # Handle key presses
            keys = []
            # Collect key presses (multiple if len(range)>1)
            for cc in range(1):
                with WithTimer('LiveVis:waitKey', quiet = self.debug_level < 2):
                    key = cv2.waitKey(self.settings.main_loop_sleep_ms)
                if key == -1:
                    break
                else:
                    keys.append(key)
                    #print 'Got key:', key
            now = time.time()
            #print 'Since last:', now - last_render

            skip_imshow = False
            #if now - last_render > .05 and since_imshow < 1:
            #    skip_imshow = True
            
            if skip_imshow:
                since_imshow += 1
            else:
                since_imshow = 0
                last_render = now

            #print '                                                         Number of keys:', len(keys)
            for key in keys:
                since_keypress = 0
                #print 'Got Key:', key
                key,do_redraw = self.handle_key_pre_apps(key)
                redraw_needed |= do_redraw
                imshow_needed |= do_redraw
                for app_name, app in self.apps.iteritems():
                    with WithTimer('%s:handle_key' % app_name, quiet = self.debug_level < 1):
                        key = app.handle_key(key, self.panes)
                key = self.handle_key_post_apps(key)
                if self.quit:
                    break
            for app_name, app in self.apps.iteritems():
                redraw_needed |= app.redraw_needed()

            # Grab latest frame from input_updater thread
            fr_idx,fr_data = self.input_updater.get_frame()
            is_new_frame = (fr_idx != latest_frame_idx and fr_data is not None)
            if is_new_frame:
                latest_frame_idx = fr_idx
                latest_frame_data = fr_data
                frame_for_apps = fr_data

            if is_new_frame:
                with WithTimer('LiveVis.display_frame', quiet = self.debug_level < 1):
                    self.display_frame(latest_frame_data)
                imshow_needed = True

            do_handle_input = (ii == 0 or
                               since_keypress >= self.settings.keypress_pause_handle_iterations)
            if frame_for_apps is not None and do_handle_input:
                # Pass frame to apps for processing
                for app_name, app in self.apps.iteritems():
                    with WithTimer('%s:handle_input' % app_name, quiet = self.debug_level < 1):
                        app.handle_input(latest_frame_data, self.panes)
                frame_for_apps = None

            # Tell each app to draw
            do_redraw = (redraw_needed and
                         (since_keypress >= self.settings.keypress_pause_redraw_iterations or
                          since_redraw >= self.settings.redraw_at_least_every))
            if redraw_needed and do_redraw:
                for app_name, app in self.apps.iteritems():
                    with WithTimer('%s:draw' % app_name, quiet = self.debug_level < 1):
                        imshow_needed |= app.draw(self.panes)
                redraw_needed = False
                since_redraw = 0

            # Render buffer
            if imshow_needed:
                with WithTimer('LiveVis:imshow', quiet = self.debug_level < 1):
                    if self.help_mode:
                        # Copy main buffer to help buffer
                        self.help_buffer[:] = self.window_buffer[:]
                        self.draw_help()
                        cv2_imshow_rgb(self.window_name, self.help_buffer)
                    else:
                        cv2_imshow_rgb(self.window_name, self.window_buffer)
                    imshow_needed = False

            ii += 1
            since_keypress += 1
            since_redraw += 1
            if ii % 2 == 0:
                sys.stdout.write('.')
            sys.stdout.flush()
            # Extra sleep just for debugging. In production all main loop sleep should be in cv2.waitKey.
            #time.sleep(2)

        print '\n\nTrying to exit run_loop...'
        self.input_updater.quit = True
        self.input_updater.join(.01 + float(self.settings.input_updater_sleep_after_read_frame) * 5)
        if self.input_updater.is_alive():
            raise Exception('Could not join self.input_updater thread')
        else:
            self.input_updater.free_camera()

        for app_name, app in self.apps.iteritems():
            print 'Quitting app:', app_name
            app.quit()

        print 'Input thread joined and apps quit; exiting run_loop.'
    
    def handle_key_pre_apps(self, key):
        tag = self.bindings.get_tag(key)
        if tag == 'freeze_cam':
            self.input_updater.freeze_cam = not self.input_updater.freeze_cam
        elif tag == 'toggle_input_mode':
            self.input_updater.toggle_input_mode()
        elif tag == 'static_file_increment':
            if self.input_updater.static_file_mode:
                self.input_updater.increment_static_file_idx(1)
            else:
                self.input_updater.static_file_mode = True
        elif tag == 'static_file_decrement':
            if self.input_updater.static_file_mode:
                self.input_updater.increment_static_file_idx(-1)
            else:
                self.input_updater.static_file_mode = True
        elif tag == 'help_mode':
            self.help_mode = not self.help_mode
        elif tag == 'stretch_mode':
            self.input_updater.toggle_stretch_mode()
            print 'Stretch mode is now', self.input_updater.static_file_stretch_mode
        elif tag == 'debug_level':
            self.debug_level = (self.debug_level + 1) % 3
            for app_name, app in self.apps.iteritems():
                app.set_debug(self.debug_level)
        else:
            return key, False
        return None, True

    def handle_key_post_apps(self, key):
        tag = self.bindings.get_tag(key)
        if tag == 'quit':
            self.quit = True
        elif key == None:
            pass
        else:
            key_label, masked_vals = self.bindings.get_key_label_from_keycode(key, extra_info = True)
            masked_vals_pp = ', '.join(['%d (%s)' % (mv, hex(mv)) for mv in masked_vals])
            if key_label is None:
                print 'Got key code %d (%s), did not match any known key (masked vals tried: %s)' % (key, hex(key), masked_vals_pp)
            elif tag is None:
                print 'Got key code %d (%s), matched key "%s", but key is not bound to any function' % (key, hex(key), key_label)
            else:
                print 'Got key code %d (%s), matched key "%s", bound to "%s", but nobody handled "%s"' % (
                    key, hex(key), key_label, tag, tag)

    def display_frame(self, frame):
        frame_disp = cv2.resize(frame[:], self.panes['input'].data.shape[:2][::-1])
        self.panes['input'].data[:] = frame_disp

    def draw_help(self):
        self.help_buffer[:] *= .7
        self.help_pane.data *= .7
        
        defaults = {'face': getattr(cv2, self.settings.caffevis_help_face),
                    'fsize': self.settings.caffevis_help_fsize,
                    'clr': to_255(self.settings.caffevis_help_clr),
                    'thick': self.settings.caffevis_help_thick}
        loc = self.settings.caffevis_help_loc[::-1]   # Reverse to OpenCV c,r order

        lines = []
        lines.append([FormattedString('~ ~ ~ Deep Visualization Toolbox ~ ~ ~', defaults, align='center', width=self.help_pane.j_size)])
        lines.append([FormattedString('', defaults)])
        lines.append([FormattedString('Base keys', defaults)])

        for tag in ('help_mode', 'freeze_cam', 'toggle_input_mode', 'static_file_increment', 'static_file_decrement', 'stretch_mode', 'quit'):
            key_strings, help_string = self.bindings.get_key_help(tag)
            label = '%10s:' % (','.join(key_strings))
            lines.append([FormattedString(label, defaults, width=120, align='right'),
                          FormattedString(help_string, defaults)])

        locy = cv2_typeset_text(self.help_pane.data, lines, loc,
                                line_spacing = self.settings.caffevis_help_line_spacing)

        for app_name, app in self.apps.iteritems():
            locy = app.draw_help(self.help_pane, locy)
