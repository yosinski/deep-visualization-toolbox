# Define critical settings and/or override defaults specified in
# settings.py. Copy this file to settings_local.py in the same
# directory as settings.py and edit. Any settings defined here
# will override those defined in settings.py

import sys
sys.dont_write_bytecode = True
from settings_local_caffenet_yos import *
#from settings_local_squeezenet import *
#from settings_local_bvlc_googlenet import *
sys.dont_write_bytecode = False



######## Extra settings for upconv
# Path to caffe deploy prototxt file. Minibatch size should be 1.
caffevis_upconv_deploy_prototxt = '/home/jason/s_local/synthesizing/nets/upconv/fc6/generator.prototxt'

# Path to network weights to load.
caffevis_upconv_network_weights = '/home/jason/s_local/synthesizing/nets/upconv/fc6/generator.caffemodel'
######## Extra settings for upconv


window_panes = (
    # (i, j, i_size, j_size)
    ('input',            (  0,    0,  300,   300)),    # This pane is required to show the input picture
    ('caffevis_aux',     (300,    0,  300,   300)),
    ('caffevis_back',    (600,    0,  300,   300)),
    ('caffevis_status',  (900,    0,   30,  1500)),
    ('caffevis_control', (  0,  300,   30,   900)),
    ('caffevis_layers',  ( 30,  300,  870,   900)),
    ('caffevis_jpgvis',  (  0, 1200,  900,   300)),
    ('caffevis_upconv',  (  0, 1500,  300,   300)),
)



    
# Set this to point to your compiled checkout of caffe
#caffevis_caffe_root      = '/Users/jason/s/caffejunk'
#caffevis_caffe_root      = '/Users/jason/s/deep-visualization-toolbox/compiled-caffe-cudnn'
#caffevis_caffe_root      = '/home/jason/s_local/deep-visualization-toolbox/compiled-caffe-cpu'
caffevis_caffe_root      = '/home/jason/s_local/deep-visualization-toolbox/compiled-caffe-cuda'
#caffevis_caffe_root      = '/home/jason/s_local/deep-visualization-toolbox/compiled-caffe-cudnn'

# Use GPU? Default is True.
caffevis_mode_gpu = True
# Display tweaks.
# Scale all window panes in UI by this factor
global_scale = .9
# Scale all fonts by this factor
global_font_size = .9
