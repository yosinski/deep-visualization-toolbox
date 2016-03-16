# Settings for Deep Visualization Toolbox
#
# Note: Probably don't change anything in this file. To override
# settings, define them in settings_local.py rather than changing them
# here.

import os
import sys

# Import local / overridden settings. Turn off creation of settings_local.pyc to avoid stale settings if settings_local.py is removed.
sys.dont_write_bytecode = True
try:
    from settings_local import *
except ImportError:
    if not os.path.exists('settings_local.py'):
        raise Exception('Could not import settings_local. Did you create it from the template? See README and start with:\n\n  $ cp models/caffenet-yos/settings_local.template-caffenet-yos.py settings_local.py')
    else:
        raise
# Resume usual pyc creation
sys.dont_write_bytecode = False



####################################
#
#  General settings
#
####################################

# Which device to use for webcam input. On Mac the default device, 0,
# works for builtin camera or external USB webcam, if plugged in. If
# you have multiple cameras, you might need to update this value. To
# disable webcam input, set to None.
input_updater_capture_device = locals().get('input_updater_capture_device', 0)

# How long to sleep in the input reading thread after reading a frame from the camera
input_updater_sleep_after_read_frame = locals().get('input_updater_sleep_after_read_frame', 1.0/20)

# Input updater thread die after this many seconds without a heartbeat. Useful during debugging to avoid other threads running after main thread has crashed.
input_updater_heartbeat_required = locals().get('input_updater_heartbeat_required', 15.0)

# How long to sleep while waiting for key presses and redraws. Recommendation: 1 (min: 1)
main_loop_sleep_ms = locals().get('main_loop_sleep_ms', 1)

# Whether or not to print a "." every second time through the main loop to visualize the loop rate
print_dots = locals().get('print_dots', False)



####################################
#
#  Window pane layout and colors/fonts
#
####################################

# Show border for each panel and annotate each with its name. Useful
# for debugging window_panes arrangement.
debug_window_panes = locals().get('debug_window_panes', False)

# The window panes available and their layout is determined by the
# "window_panes" variable. By default all panes are enabled with a
# standard size. This setting will often be overridden on a per-model
# basis, e.g. if the model does not have pre-computed jpgvis
# information, the caffevis_jpgvis pane can be omitted. For
# convenience, if the only variable that needs to be overridden is the
# height of the control panel (to accomodate varying length of layer
# names), one can simply define control_pane_height. If more
if 'default_window_panes' in locals():
    raise Exception('Override window panes in settings_local.py by defining window_panes, not default_window_panes')
default_window_panes = (
    # (i, j, i_size, j_size)
    ('input',            (  0,    0,  300,   300)),    # This pane is required to show the input picture
    ('caffevis_aux',     (300,    0,  300,   300)),
    ('caffevis_back',    (600,    0,  300,   300)),
    ('caffevis_status',  (900,    0,   30,  1500)),
    ('caffevis_control', (  0,  300,   30,   900)),
    ('caffevis_layers',  ( 30,  300,  870,   900)),
    ('caffevis_jpgvis',  (  0, 1200,  900,   300)),
)
window_panes = locals().get('window_panes', default_window_panes)

# Define global_scale as a float to rescale window and all
# panes. Handy for quickly changing resolution for a different screen.
global_scale = locals().get('global_scale', 1.0)

# Define global_font_size to scale all font sizes by this amount.
global_font_size = locals().get('global_font_size', 1.0)

if global_scale != 1.0:
    scaled_window_panes = []
    for wp in window_panes:
        scaled_window_panes.append([wp[0], [int(val*global_scale) for val in wp[1]]])
    window_panes = scaled_window_panes

# All window configuation information is now contained in the
# window_panes variable. Print if desired:
if debug_window_panes:
    print 'Final window panes and locations/sizes (i, j, i_size, j_size):'
    for pane in window_panes:
        print '  Pane: %s' % repr(pane)

help_pane_loc = locals().get('help_pane_loc', (.07, .07, .86, .86))    # as a fraction of main window
window_background = locals().get('window_background', (.2, .2, .2))
stale_background =locals().get('stale_background',  (.3, .3, .2))
static_files_dir = locals().get('static_files_dir', 'input_images')
static_files_regexp = locals().get('static_files_regexp', '.*\.(jpg|jpeg|png)$')
static_files_ignore_case = locals().get('static_files_ignore_case', True)
# True to stretch to square, False to crop to square. (Can change at
# runtime via 'stretch_mode' key.)
static_file_stretch_mode = locals().get('static_file_stretch_mode', False)

# int, 0+. How many times to go through the main loop after a keypress
# before resuming handling frames (0 to handle every frame as it
# arrives). Setting this to a value > 0 can enable more responsive
# keyboard input even when other settings are tuned to maximize the
# framerate. Default: 2
keypress_pause_handle_iterations = locals().get('keypress_pause_handle_iterations', 2)

# int, 0+. How many times to go through the main loop after a keypress
# before resuming redraws (0 to redraw every time it is
# needed). Setting this to a value > 0 can enable more responsive
# keyboard input even when other settings are tuned to maximize the
# framerate. Default: 1
keypress_pause_redraw_iterations = locals().get('keypress_pause_redraw_iterations', 1)

# int, 1+. Force a redraw even when keys are pressed if there have
# been this many passes through the main loop without a redraw due to
# the keypress_pause_redraw_iterations setting combined with many key
# presses. Default: 3.
redraw_at_least_every = locals().get('redraw_at_least_every', 3)

# Tuple of tuples describing the file to import and class from it to
# instantiate for each app to be run. Apps are run and given keys to
# handle in the order specified.
default_installed_apps = (
    ('caffevis.app', 'CaffeVisApp'),
)
installed_apps = locals().get('installed_apps', default_installed_apps)

# Font settings for the help pane. Text is rendered using OpenCV; see
# http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html#puttext
# for information on parameters.
help_face = locals().get('help_face', 'FONT_HERSHEY_COMPLEX_SMALL')
help_loc = locals().get('help_loc', (20,10))   # r,c order
help_line_spacing = locals().get('help_line_spacing', 10)     # extra pixel spacing between lines
help_clr   = locals().get('help_clr', (1,1,1))
help_fsize = locals().get('help_fsize', 1.0 * global_font_size)
help_thick = locals().get('help_thick', 1)



####################################
#
#  Caffevis settings
#
####################################

# Whether to use GPU mode (if True) or CPU mode (if False)
caffevis_mode_gpu = locals().get('caffevis_mode_gpu', True)

# Data mean, if any, to be subtracted from input image file / webcam
# image. Specify as string path to file or tuple of one value per
# channel or None.
caffevis_data_mean = locals().get('caffevis_data_mean', None)

# Path to file listing labels in order, one per line, used for the
# below two features. None to disable.
caffevis_labels = locals().get('caffevis_labels', None)

# Which layers have channels/neurons corresponding to the order given
# in the caffevis_labels file? Annotate these units with label text
# (when those neurons are selected). None to disable.
caffevis_label_layers = locals().get('caffevis_label_layers', None)

# Which layer to use for displaying class output numbers in left pane
# (when no neurons are selected). None to disable.
caffevis_prob_layer = locals().get('caffevis_prob_layer', None)

# String or None. Which directory to load pre-computed per-unit
# visualizations from, if any. None to disable.
caffevis_unit_jpg_dir = locals().get('caffevis_unit_jpg_dir', None)

# List. For which layers should jpgs be loaded for
# visualization? If a layer name (full name, not prettified) is given
# here, we will try to load jpgs to visualize each unit. This is used
# for pattern mode ('s' key by default) and for the right
# caffevis_jpgvis pane ('9' key by default). Empty list to disable.
caffevis_jpgvis_layers = locals().get('caffevis_jpgvis_layers', [])

# Dict specifying string:string mapping. Steal pattern mode and right
# jpgvis pane visualizations for certain layers (e.g. pool1) from
# other layers (e.g. conv1). We can do this because
# optimization/max-act/deconv-of-max results are identical.
caffevis_jpgvis_remap = locals().get('caffevis_jpgvis_remap', {})

# Function mapping old name -> new name to modify/prettify/shorten
# layer names.
caffevis_layer_pretty_name_fn = locals().get('caffevis_layer_pretty_name_fn', lambda name: name)

# The CaffeVisApp computes a layout of neurons for the caffevis_layers
# pane given the aspect ratio in caffevis_layers_aspect_ratio (< 1 for
# portrait, 1 for square, > 1 for landscape). Default: 1 (square).
caffevis_layers_aspect_ratio = locals().get('caffevis_layers_aspect_ratio', 1.0)

# Replace magic '%DVT_ROOT%' string with the root DeepVis Toolbox
# directory (the location of this settings file)
dvt_root = os.path.dirname(os.path.abspath(__file__))
if 'caffevis_deploy_prototxt' in locals():
    caffevis_deploy_prototxt = caffevis_deploy_prototxt.replace('%DVT_ROOT%', dvt_root)
if 'caffevis_network_weights' in locals():
    caffevis_network_weights = caffevis_network_weights.replace('%DVT_ROOT%', dvt_root)
if isinstance(caffevis_data_mean, basestring):
    caffevis_data_mean = caffevis_data_mean.replace('%DVT_ROOT%', dvt_root)
if isinstance(caffevis_labels, basestring):
    caffevis_labels = caffevis_labels.replace('%DVT_ROOT%', dvt_root)
if isinstance(caffevis_unit_jpg_dir, basestring):
    caffevis_unit_jpg_dir = caffevis_unit_jpg_dir.replace('%DVT_ROOT%', dvt_root)

# Pause Caffe forward/backward computation for this many seconds after a keypress. This is to keep the processor free for a brief period after a keypress, which allow the interface to feel much more responsive. After this period has passed, Caffe resumes computation, in CPU mode often occupying all cores. Default: .1
caffevis_pause_after_keys = locals().get('caffevis_pause_after_keys', .10)
caffevis_frame_wait_sleep = locals().get('caffevis_frame_wait_sleep', .01)
caffevis_jpg_load_sleep = locals().get('caffevis_jpg_load_sleep', .01)
# CaffeProc thread dies after this many seconds without a
# heartbeat. Useful during debugging to avoid other threads running
# after main thread has crashed.
caffevis_heartbeat_required = locals().get('caffevis_heartbeat_required', 15.0)

# How far to move when using fast left/right/up/down keys
caffevis_fast_move_dist = locals().get('caffevis_fast_move_dist', 3)
# Size of jpg reading cache in bytes (default: 2GB)
# Note: largest fc6/fc7 images are ~600MB. Cache smaller than this will be painfully slow when using patterns_mode for fc6 and fc7.
# Cache use when all layers have been loaded is ~1.6GB
caffevis_jpg_cache_size  = locals().get('caffevis_jpg_cache_size', 2000*1024**2)

caffevis_grad_norm_blur_radius = locals().get('caffevis_grad_norm_blur_radius', 4.0)

# Boost display of individual channels. For channel activations in the
# range [0,1], boost_indiv rescales the activations of that channel
# such that the new_max = old_max ** -boost_indiv. Thus no-op value =
# 0.0, and a value of 1.0 means each channel is scaled to use the
# entire [0,1] range.
caffevis_boost_indiv_choices = locals().get('caffevis_boost_indiv_choices', (0, .3, .5, .8, 1))
# Default boost indiv given as index into caffevis_boost_indiv_choices
caffevis_boost_indiv_default_idx = locals().get('caffevis_boost_indiv_default_idx', 0)
# Boost display of entire layer activation by the given gamma value
# (for values in [0,1], display_val = old_val ** gamma. No-op value:
# 1.0)
caffevis_boost_gamma_choices = locals().get('caffevis_boost_gamma_choices', (1, .7, .5, .3))
# Default boost gamma given as index into caffevis_boost_gamma_choices
caffevis_boost_gamma_default_idx = locals().get('caffevis_boost_gamma_default_idx', 0)
# Initially show label predictions or not (toggle with default key '8')
caffevis_init_show_label_predictions = locals().get('caffevis_init_show_label_predictions', True)
# Initially show jpg vis or not (toggle with default key '9')
caffevis_init_show_unit_jpgs = locals().get('caffevis_init_show_unit_jpgs', True)

# extra pixel spacing between lines. Default: 4 = not much space / tight layout
caffevis_control_line_spacing = locals().get('caffevis_control_line_spacing', 4)
# Font settings for control pane (list of layers)
caffevis_control_face = locals().get('caffevis_control_face', 'FONT_HERSHEY_COMPLEX_SMALL')
caffevis_control_loc = locals().get('caffevis_control_loc', (15,5))   # r,c order
caffevis_control_clr = locals().get('caffevis_control_clr', (.8,.8,.8))
caffevis_control_clr_selected = locals().get('caffevis_control_clr_selected', (1, 1, 1))
caffevis_control_clr_cursor = locals().get('caffevis_control_clr_cursor', (.5,1,.5))
caffevis_control_clr_bp = locals().get('caffevis_control_clr_bp', (.8, .8, 1))
caffevis_control_fsize = locals().get('caffevis_control_fsize', 1.0 * global_font_size)
caffevis_control_thick = locals().get('caffevis_control_thick', 1)
caffevis_control_thick_selected = locals().get('caffevis_control_thick_selected', 2)
caffevis_control_thick_cursor = locals().get('caffevis_control_thick_cursor', 2)
caffevis_control_thick_bp = locals().get('caffevis_control_thick_bp', 2)

# Color settings for layer activation pane
caffevis_layer_clr_cursor   = locals().get('caffevis_layer_clr_cursor', (.5,1,.5))
caffevis_layer_clr_back_background = locals().get('caffevis_layer_clr_back_background', (.2,.2,.5))
caffevis_layer_clr_back_sel = locals().get('caffevis_layer_clr_back_sel', (.2,.2,1))

# Font settings for status pane (bottom line)
caffevis_status_face = locals().get('caffevis_status_face', 'FONT_HERSHEY_COMPLEX_SMALL')
caffevis_status_loc = locals().get('caffevis_status_loc', (15,10))   # r,c order
caffevis_status_line_spacing = locals().get('caffevis_status_line_spacing', 5)     # extra pixel spacing between lines
caffevis_status_clr = locals().get('caffevis_status_clr', (.8,.8,.8))
caffevis_status_fsize = locals().get('caffevis_status_fsize', 1.0 * global_font_size)
caffevis_status_thick = locals().get('caffevis_status_thick', 1)
caffevis_jpgvis_stack_vert = locals().get('caffevis_jpgvis_stack_vert', True)

# Font settings for class prob output (top 5 classes listed on left)
caffevis_class_face = locals().get('caffevis_class_face', 'FONT_HERSHEY_COMPLEX_SMALL')
caffevis_class_loc = locals().get('caffevis_class_loc', (20,10))   # r,c order
caffevis_class_line_spacing = locals().get('caffevis_class_line_spacing', 10)     # extra pixel spacing between lines
caffevis_class_clr_0 = locals().get('caffevis_class_clr_0', (.5,.5,.5))
caffevis_class_clr_1 = locals().get('caffevis_class_clr_1', (.5,1,.5))
caffevis_class_fsize = locals().get('caffevis_class_fsize', 1.0 * global_font_size)
caffevis_class_thick = locals().get('caffevis_class_thick', 1)

# Font settings for label overlay text (shown on layer pane only for caffevis_label_layers layers)
caffevis_label_face = locals().get('caffevis_label_face', 'FONT_HERSHEY_COMPLEX_SMALL')
caffevis_label_loc = locals().get('caffevis_label_loc', (30,20))   # r,c order
caffevis_label_clr = locals().get('caffevis_label_clr', (.8,.8,.8))
caffevis_label_fsize = locals().get('caffevis_label_fsize', 1.0 * global_font_size)
caffevis_label_thick = locals().get('caffevis_label_thick', 1)



####################################
#
#  A few final sanity checks
#
####################################

# Check that required setting have been defined
bound_locals = locals()
def assert_in_settings(setting_name):
    if not setting_name in bound_locals:
        raise Exception('The "%s" setting is required; be sure to define it in settings_local.py' % setting_name)

assert_in_settings('caffevis_caffe_root')
assert_in_settings('caffevis_deploy_prototxt')
assert_in_settings('caffevis_network_weights')
assert_in_settings('caffevis_data_mean')

# Check that caffe directory actually exists
if not os.path.exists(caffevis_caffe_root):
    raise Exception('The Caffe directory specified in settings_local.py, %s, does not exist. Set the caffevis_caffe_root variable in your settings_local.py to the path of your compiled Caffe checkout.' % caffevis_caffe_root)
