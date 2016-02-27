import numpy as np

from image_misc import get_tiles_height_width, caffe_load_image



def net_preproc_forward(net, img, data_hw):
    appropriate_shape = data_hw + (3,)
    assert img.shape == appropriate_shape, 'img is wrong size (got %s but expected %s)' % (img.shape, appropriate_shape)
    #resized = caffe.io.resize_image(img, net.image_dims)   # e.g. (227, 227, 3)
    data_blob = net.transformer.preprocess('data', img)                # e.g. (3, 227, 227), mean subtracted and scaled to [0,255]
    data_blob = data_blob[np.newaxis,:,:,:]                   # e.g. (1, 3, 227, 227)
    output = net.forward(data=data_blob)
    return output


def get_pretty_layer_name(settings, layer_name):
    has_old_settings = hasattr(settings, 'caffevis_layer_pretty_names')
    has_new_settings = hasattr(settings, 'caffevis_layer_pretty_name_fn')
    if has_old_settings and not has_new_settings:
        print ('WARNING: Your settings.py and/or settings_local.py are out of date.'
               'caffevis_layer_pretty_names has been replaced with caffevis_layer_pretty_name_fn.'
               'Update your settings.py and/or settings_local.py (see documentation in'
               'setttings.py) to remove this warning.')
        return settings.caffevis_layer_pretty_names.get(layer_name, layer_name)

    ret = layer_name
    if hasattr(settings, 'caffevis_layer_pretty_name_fn'):
        ret = settings.caffevis_layer_pretty_name_fn(ret)
    if ret != layer_name:
        print '  Prettified layer name: "%s" -> "%s"' % (layer_name, ret)
    return ret


def read_label_file(filename):
    ret = []
    with open(filename, 'r') as ff:
        for line in ff:
            label = line.strip()
            if len(label) > 0:
                ret.append(label)
    return ret


def crop_to_corner(img, corner, small_padding = 1, large_padding = 2):
    '''Given an large image consisting of 3x3 squares with small_padding padding concatenated into a 2x2 grid with large_padding padding, return one of the four corners (0, 1, 2, 3)'''
    assert corner in (0,1,2,3), 'specify corner 0, 1, 2, or 3'
    assert img.shape[0] == img.shape[1], 'img is not square'
    assert img.shape[0] % 2 == 0, 'even number of pixels assumption violated'
    half_size = img.shape[0]/2
    big_ii = 0 if corner in (0,1) else 1
    big_jj = 0 if corner in (0,2) else 1
    tp = small_padding + large_padding
    #tp = 0
    return img[big_ii*half_size+tp:(big_ii+1)*half_size-tp,
               big_jj*half_size+tp:(big_jj+1)*half_size-tp]


def load_sprite_image(img_path, rows_cols, n_sprites = None):
    '''Load a 2D (3D with color channels) sprite image where
    (rows,cols) = rows_cols, slices, and returns as a 3D tensor (4D
    with color channels). Sprite shape is computed automatically. If
    n_sprites is not given, it is assumed to be rows*cols. Return as
    3D tensor with shape (n_sprites, sprite_height, sprite_width,
    sprite_channels).
    '''

    rows,cols = rows_cols
    if n_sprites is None:
        n_sprites = rows * cols
    img = caffe_load_image(img_path, color = True, as_uint = True)
    assert img.shape[0] % rows == 0, 'sprite image has shape %s which is not divisible by rows_cols %' % (img.shape, rows_cols)
    assert img.shape[1] % cols == 0, 'sprite image has shape %s which is not divisible by rows_cols %' % (img.shape, rows_cols)
    sprite_height = img.shape[0] / rows
    sprite_width  = img.shape[1] / cols
    sprite_channels = img.shape[2]

    ret = np.zeros((n_sprites, sprite_height, sprite_width, sprite_channels), dtype = img.dtype)
    for idx in xrange(n_sprites):
        # Row-major order
        ii = idx / cols
        jj = idx % cols
        ret[idx] = img[ii*sprite_height:(ii+1)*sprite_height,
                       jj*sprite_width:(jj+1)*sprite_width, :]
    return ret


def load_square_sprite_image(img_path, n_sprites):
    '''
    Just like load_sprite_image but assumes tiled image is square
    '''
    
    tile_rows,tile_cols = get_tiles_height_width(n_sprites)
    return load_sprite_image(img_path, (tile_rows, tile_cols), n_sprites = n_sprites)


def check_force_backward_true(prototxt_file):
    '''Checks whether the given file contains a line with the following text, ignoring whitespace:
    force_backward: true
    '''

    found = False
    with open(prototxt_file, 'r') as ff:
        for line in ff:
            fields = line.strip().split()
            if len(fields) == 2 and fields[0] == 'force_backward:' and fields[1] == 'true':
                found = True
                break

    if not found:
        print '\n\nWARNING: the specified prototxt'
        print '"%s"' % prototxt_file
        print 'does not contain the line "force_backward: true". This may result in backprop'
        print 'and deconv producing all zeros at the input layer. You may want to add this line'
        print 'to your prototxt file before continuing to force backprop to compute derivatives'
        print 'at the data layer as well.\n\n'

