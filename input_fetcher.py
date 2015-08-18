import os
import cv2
import re
import time
from threading import RLock

from codependent_thread import CodependentThread
from image_misc import cv2_imshow_rgb, cv2_read_file_rgb, read_cam_frame, crop_to_square



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
                print '\n\nWARNING: InputImageFetcher: capture device %s failed to open! Camera will not be available!\n\n' % self.capture_device
                self.bound_cap_device = None
                self.no_cam_present = True
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
