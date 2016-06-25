# Define critical settings and/or override defaults specified in
# settings.py. Copy this file to settings_local.py in the same
# directory as settings.py and edit. Any settings defined here
# will override those defined in settings.py

# Load model: caffenet-yos
# Path to caffe deploy prototxt file. Minibatch size should be 1.
caffevis_deploy_prototxt = '%DVT_ROOT%/models/caffenet-yos/caffenet-yos-deploy-nonreused.prototxt'

# Path to network weights to load.
caffevis_network_weights = '%DVT_ROOT%/models/caffenet-yos/caffenet-yos-weights'

# Other optional settings; see complete documentation for each in settings.py.
caffevis_data_mean       = '%DVT_ROOT%/models/caffenet-yos/ilsvrc_2012_mean.npy'
caffevis_labels          = '%DVT_ROOT%/models/caffenet-yos/ilsvrc_2012_labels.txt'
caffevis_label_layers    = ('fc8', 'prob')
caffevis_prob_layer      = 'prob'
caffevis_unit_jpg_dir    = '%DVT_ROOT%/models/caffenet-yos/unit_jpg_vis'
caffevis_jpgvis_layers   = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8', 'prob']
caffevis_jpgvis_remap    = {'pool1': 'conv1', 'pool2': 'conv2', 'pool5': 'conv5',
                            'relu1': 'conv1', 'relu2': 'conv2', 'relu3': 'conv3', 'relu4': 'conv4', 'relu5': 'conv5',
                            'norm1': 'conv1', 'norm2': 'conv2',
                            'relu6': 'fc6', 'drop6': 'fc6', 'relu7': 'fc7', 'drop7': 'fc7',
                            'prob': 'fc8'}
def caffevis_layer_pretty_name_fn(name):
    return name.replace('pool','p').replace('norm','n').replace('relu','r')


######## Extra settings for upconv
# Path to caffe deploy prototxt file. Minibatch size should be 1.
caffevis_upconv_deploy_prototxt = '/home/jason/s_local/synthesizing/nets/upconv/fc6/generator.prototxt'
#caffevis_upconv_deploy_prototxt = '/home/jason/s_local/synthesizing/nets/upconv/conv2/decoder.prototxt'

# Path to network weights to load.
caffevis_upconv_network_weights = '/home/jason/s_local/synthesizing/nets/upconv/fc6/generator.caffemodel'
#caffevis_upconv_network_weights = '/home/jason/s_local/synthesizing/nets/upconv/conv2/generator.caffemodel'

# Path to range clip.
caffevis_upconv_code_clip = '/home/jason/s_local/synthesizing/act_range/3x/fc6.npy'
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
    ('caffevis_upconv',  (  0, 1510,  600,   600)),
)


# Set this to point to your compiled checkout of caffe
caffevis_caffe_root      = '/path/to/caffe'

# Use GPU? Default is True.
caffevis_mode_gpu = True
# Display tweaks.
# Scale all window panes in UI by this factor
global_scale = 1.2
# Scale all fonts by this factor
global_font_size = .9
