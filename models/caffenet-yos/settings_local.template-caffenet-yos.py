# Define critical settings and/or override defaults specified in
# settings.py. Copy this file to settings_local.py in the same
# directory as settings.py and edit. Any settings defined here
# will override those defined in settings.py



# Set this to point to your compiled checkout of caffe
caffevis_caffe_root      = '/path/to/caffe'

# Load model: caffenet-yos
# Path to caffe deploy prototxt file. Minibatch size should be 1.
caffevis_deploy_prototxt = '%DVT_ROOT%/models/caffenet-yos/caffenet-yos-deploy.prototxt'

# Path to network weights to load.
caffevis_network_weights = '%DVT_ROOT%/models/caffenet-yos/caffenet-yos-weights'

# Other optional settings; see complete documentation for each in settings.py.
caffevis_data_mean       = '%DVT_ROOT%/models/caffenet-yos/ilsvrc_2012_mean.npy'
caffevis_labels          = '%DVT_ROOT%/models/caffenet-yos/ilsvrc_2012_labels.txt'
caffevis_label_layers    = ('fc8', 'prob')
caffevis_prob_layer      = 'prob'
caffevis_unit_jpg_dir    = '%DVT_ROOT%/models/caffenet-yos/unit_jpg_vis'
caffevis_jpgvis_layers   = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8', 'prob']
caffevis_jpgvis_remap    = {'pool1': 'conv1', 'pool2': 'conv2', 'pool5': 'conv5'}
def caffevis_layer_pretty_name_fn(name):
    return name.replace('pool','p').replace('norm','n')

# Use GPU? Default is True.
#caffevis_mode_gpu = True
# Display tweaks.
# Scale all window panes in UI by this factor
#global_scale = 1.0
# Scale all fonts by this factor    
#global_font_size = 1.0
