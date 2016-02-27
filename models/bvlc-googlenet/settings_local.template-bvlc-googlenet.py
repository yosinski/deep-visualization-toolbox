# Define critical settings and/or override defaults specified in
# settings.py. Copy this file to settings_local.py in the same
# directory as settings.py and edit. Any settings defined here
# will override those defined in settings.py



# Set this to point to your compiled checkout of caffe
caffevis_caffe_root      = '/path/to/caffe'

# Load model: bvlc-googlenet
# Path to caffe deploy prototxt file. Minibatch size should be 1.
caffevis_deploy_prototxt = '%DVT_ROOT%/models/bvlc-googlenet/bvlc-googlenet-deploy.prototxt'

# Path to network weights to load.
caffevis_network_weights = '%DVT_ROOT%/models/bvlc-googlenet/bvlc-googlenet.caffemodel'

# Other optional settings; see complete documentation for each in settings.py.
caffevis_data_mean       = (104, 117, 123)   # per-channel mean
caffevis_labels          = '%DVT_ROOT%/models/bvlc-googlenet/ilsvrc_2012_labels.txt'
caffevis_jpgvis_layers   = []
caffevis_prob_layer      = 'prob'
caffevis_label_layers    = ('loss3/classifier', 'prob')
def caffevis_layer_pretty_name_fn(name):
    # Shorten many layer names to fit in control pane (full layer name visible in status bar)
    name = name.replace('conv','c').replace('pool','p').replace('norm','n')
    name = name.replace('inception_','i').replace('output','o').replace('reduce','r').replace('split_','s')
    name = name.replace('__','_').replace('__','_')
    return name
# Don't display duplicate *_split_* layers
caffevis_filter_layers = lambda name: '_split_' in name

# Window panes for bvlc-googlenet (no caffevis_jpgvis pane, larger control pane to fit many layer names).
_control_height = 125
window_panes = (
    # (i, j, i_size, j_size)
    ('input',            (  0,    0,  300,   300)),
    ('caffevis_aux',     (300,    0,  300,   300)),
    ('caffevis_back',    (600,    0,  300,   300)),
    ('caffevis_status',  (900,    0,   30,  1500)),
    ('caffevis_control', (  0,  300,   _control_height, 1200)),
    ('caffevis_layers',  ( _control_height,  300,  900 - _control_height, 1200)),
)
caffevis_layers_aspect_ratio = float(window_panes[-1][1][3])/window_panes[-1][1][2]  # Actual ratio from caffevis_layers
caffevis_control_fsize = .85

# Use GPU? Default is True.
#caffevis_mode_gpu = True
# Display tweaks.
# Scale all window panes in UI by this factor
#global_scale = 1.0
# Scale all fonts by this factor    
#global_font_size = 1.0

