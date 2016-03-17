#! /usr/bin/env python

import os
import ipdb as pdb
import errno
from datetime import datetime

#import caffe
from loaders import load_imagenet_mean, load_labels, caffe
from jby_misc import WithTimer
from caffe_misc import shownet, RegionComputer, save_caffe_image
import numpy as np



default_layers  = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8', 'prob']
default_is_conv = [('conv' in ll) for ll in default_layers]

def hardcoded_get():
    prototxt = '/home/jyosinsk/results/140311_234854_afadfd3_priv_netbase_upgraded/deploy_1.prototxt'
    weights = '/home/jyosinsk/results/140311_234854_afadfd3_priv_netbase_upgraded/caffe_imagenet_train_iter_450000'
    datadir = '/home/jyosinsk/imagenet2012/val'
    filelist = 'mini_valid.txt'

    imagenet_mean = load_imagenet_mean()
    net = caffe.Classifier(prototxt, weights,
                           mean=imagenet_mean,
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256, 256))
    net.set_phase_test()
    net.set_mode_cpu()
    labels = load_labels()

    return net, imagenet_mean, labels, datadir, filelist



def mkdir_p(path):
    # From https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise



class MaxTracker(object):
    def __init__(self, is_conv, n_channels, n_top = 10, initial_val = -1e99, dtype = 'float32'):
        self.is_conv = is_conv
        self.max_vals = np.ones((n_channels, n_top), dtype = dtype) * initial_val
        self.n_top = n_top
        if is_conv:
            self.max_locs = -np.ones((n_channels, n_top, 4), dtype = 'int')   # image_idx, image_class, i, j
        else:
            self.max_locs = -np.ones((n_channels, n_top, 2), dtype = 'int')      # image_idx, image_class

    def update(self, blob, image_idx, image_class):
        data = blob[0]                                        # Note: makes a copy of blob, e.g. (96,55,55)
        n_channels = data.shape[0]
        data_unroll = data.reshape((n_channels, -1))          # Note: no copy eg (96,3025). Does nothing if not is_conv

        maxes = data_unroll.argmax(1)   # maxes for each channel, eg. (96,)

        #insertion_idx = zeros((n_channels,))
        #pdb.set_trace()
        for ii in xrange(n_channels):
            idx = np.searchsorted(self.max_vals[ii], data_unroll[ii, maxes[ii]])
            if idx == 0:
                # Smaller than all 10
                continue
            # Store new max in the proper order. Update both arrays:
            # self.max_vals:
            self.max_vals[ii,:idx-1] = self.max_vals[ii,1:idx]       # shift lower values
            self.max_vals[ii,idx-1] = data_unroll[ii, maxes[ii]]     # store new max value
            # self.max_locs
            self.max_locs[ii,:idx-1] = self.max_locs[ii,1:idx]       # shift lower location data
            # store new location
            if self.is_conv:
                self.max_locs[ii,idx-1] = (image_idx, image_class) + np.unravel_index(maxes[ii], data.shape[1:])
            else:
                self.max_locs[ii,idx-1] = (image_idx, image_class)



class NetMaxTracker(object):
    def __init__(self, layers = default_layers, is_conv = default_is_conv, n_top = 10, initial_val = -1e99, dtype = 'float32'):
        self.layers = layers
        self.is_conv = is_conv
        self.init_done = False
        self.n_top = n_top
        self.initial_val = initial_val

    def _init_with_net(self, net):
        self.max_trackers = {}

        for layer,is_conv in zip(self.layers, self.is_conv):
            blob = net.blobs[layer].data
            self.max_trackers[layer] = MaxTracker(is_conv, blob.shape[1], n_top = self.n_top,
                                                  initial_val = self.initial_val,
                                                  dtype = blob.dtype)
        self.init_done = True
        
    def update(self, net, image_idx, image_class):
        '''Updates the maxes found so far with the state of the given net. If a new max is found, it is stored together with the image_idx.'''
        if not self.init_done:
            self._init_with_net(net)

        for layer in self.layers:
            blob = net.blobs[layer].data
            self.max_trackers[layer].update(blob, image_idx, image_class)


def load_file_list(filelist):
    image_filenames = []
    image_labels = []
    with open(filelist, 'r') as ff:
        for line in ff.readlines():
            fields = line.strip().split()
            image_filenames.append(fields[0])
            image_labels.append(int(fields[1]))
    return image_filenames, image_labels



def scan_images_for_maxes(net, datadir, filelist, n_top):
    image_filenames, image_labels = load_file_list(filelist)
    print 'Scanning %d files' % len(image_filenames)
    print '  First file', os.path.join(datadir, image_filenames[0])

    tracker = NetMaxTracker(n_top = n_top)
    for image_idx in xrange(len(image_filenames)):
        filename = image_filenames[image_idx]
        image_class = image_labels[image_idx]
        #im = caffe.io.load_image('../../data/ilsvrc12/mini_ilsvrc_valid/sized/ILSVRC2012_val_00000610.JPEG')
        do_print = (image_idx % 100 == 0)
        if do_print:
            print '%s   Image %d/%d' % (datetime.now().ctime(), image_idx, len(image_filenames))
        with WithTimer('Load image', quiet = not do_print):
            im = caffe.io.load_image(os.path.join(datadir, filename))
        with WithTimer('Predict   ', quiet = not do_print):
            net.predict([im], oversample = False)   # Just take center crop
        with WithTimer('Update    ', quiet = not do_print):
            tracker.update(net, image_idx, image_class)

    print 'done!'
    return tracker



def save_representations(net, datadir, filelist, layer, first_N = None):
    image_filenames, image_labels = load_file_list(filelist)
    if first_N is None:
        first_N = len(image_filenames)
    assert first_N <= len(image_filenames)
    image_indices = range(first_N)
    print 'Scanning %d files' % len(image_indices)
    assert len(image_indices) > 0
    print '  First file', os.path.join(datadir, image_filenames[image_indices[0]])

    indices = None
    rep = None
    for ii,image_idx in enumerate(image_indices):
        filename = image_filenames[image_idx]
        image_class = image_labels[image_idx]
        do_print = (image_idx % 10 == 0)
        if do_print:
            print '%s   Image %d/%d' % (datetime.now().ctime(), image_idx, len(image_indices))
        with WithTimer('Load image', quiet = not do_print):
            im = caffe.io.load_image(os.path.join(datadir, filename))
        with WithTimer('Predict   ', quiet = not do_print):
            net.predict([im], oversample = False)   # Just take center crop
        with WithTimer('Store     ', quiet = not do_print):
            if rep is None:
                rep_shape = net.blobs[layer].data[0].shape   # e.g. (256,13,13)
                rep = np.zeros((len(image_indices),) + rep_shape)   # e.g. (1000,256,13,13)
                indices = [0] * len(image_indices)
            indices[ii] = image_idx
            rep[ii] = net.blobs[layer].data[0]

    print 'done!'
    return indices,rep



def get_max_data_extent(net, layer, rc, is_conv):
    '''Gets the maximum size of the data layer that can influence a unit on layer.'''
    if is_conv:
        conv_size = net.blobs[layer].data.shape[2:4]        # e.g. (13,13) for conv5
        layer_slice_middle = (conv_size[0]/2,conv_size[0]/2+1, conv_size[1]/2,conv_size[1]/2+1)   # e.g. (6,7,6,7,), the single center unit
        data_slice = rc.convert_region(layer, 'data', layer_slice_middle)
        return data_slice[1]-data_slice[0], data_slice[3]-data_slice[2]   # e.g. (163, 163) for conv5
    else:
        # Whole data region
        return net.blobs['data'].data.shape[2:4]        # e.g. (227,227) for fc6,fc7,fc8,prop



def output_max_patches(max_tracker, net, layer, idx_begin, idx_end, num_top, datadir, filelist, outdir, do_which):
    do_maxes, do_deconv, do_deconv_norm, do_backprop, do_backprop_norm, do_info = do_which
    assert do_maxes or do_deconv or do_deconv_norm or do_backprop or do_backprop_norm or do_info, 'nothing to do'

    mt = max_tracker
    rc = RegionComputer()
    
    image_filenames, image_labels = load_file_list(filelist)
    print 'Loaded filenames and labels for %d files' % len(image_filenames)
    print '  First file', os.path.join(datadir, image_filenames[0])

    num_top_in_mt = mt.max_locs.shape[1]
    assert num_top <= num_top_in_mt, 'Requested %d top images but MaxTracker contains only %d' % (num_top, num_top_in_mt)
    assert idx_end >= idx_begin, 'Range error'

    size_ii, size_jj = get_max_data_extent(net, layer, rc, mt.is_conv)
    data_size_ii, data_size_jj = net.blobs['data'].data.shape[2:4]
    
    n_total_images = (idx_end-idx_begin) * num_top
    for cc, channel_idx in enumerate(range(idx_begin, idx_end)):
        unit_dir = os.path.join(outdir, layer, 'unit_%04d' % channel_idx)
        mkdir_p(unit_dir)

        if do_info:
            info_filename = os.path.join(unit_dir, 'info.txt')
            info_file = open(info_filename, 'w')
            print >>info_file, '# is_conv val image_idx image_class i(if is_conv) j(if is_conv) filename'

        # iterate through maxes from highest (at end) to lowest
        for max_idx_0 in range(num_top):
            max_idx = num_top_in_mt - 1 - max_idx_0
            if mt.is_conv:
                im_idx, im_class, ii, jj = mt.max_locs[channel_idx, max_idx]
            else:
                im_idx, im_class = mt.max_locs[channel_idx, max_idx]
            recorded_val = mt.max_vals[channel_idx, max_idx]
            filename = image_filenames[im_idx]
            do_print = (max_idx_0 == 0)
            if do_print:
                print '%s   Output file/image(s) %d/%d' % (datetime.now().ctime(), cc * num_top, n_total_images)

            if mt.is_conv:
                # Compute the focus area of the data layer
                layer_indices = (ii,ii+1,jj,jj+1)
                data_indices = rc.convert_region(layer, 'data', layer_indices)
                data_ii_start,data_ii_end,data_jj_start,data_jj_end = data_indices

                touching_imin = (data_ii_start == 0)
                touching_jmin = (data_jj_start == 0)

                # Compute how much of the data slice falls outside the actual data [0,max] range
                ii_outside = size_ii - (data_ii_end - data_ii_start)     # possibly 0
                jj_outside = size_jj - (data_jj_end - data_jj_start)     # possibly 0

                if touching_imin:
                    out_ii_start = ii_outside
                    out_ii_end   = size_ii
                else:
                    out_ii_start = 0
                    out_ii_end   = size_ii - ii_outside
                if touching_jmin:
                    out_jj_start = jj_outside
                    out_jj_end   = size_jj
                else:
                    out_jj_start = 0
                    out_jj_end   = size_jj - jj_outside
            else:
                ii,jj = 0,0
                data_ii_start, out_ii_start, data_jj_start, out_jj_start = 0,0,0,0
                data_ii_end, out_ii_end, data_jj_end, out_jj_end = size_ii, size_ii, size_jj, size_jj


            if do_info:
                print >>info_file, 1 if mt.is_conv else 0, '%.6f' % mt.max_vals[channel_idx, max_idx],
                if mt.is_conv:
                    print >>info_file, '%d %d %d %d' % tuple(mt.max_locs[channel_idx, max_idx]),
                else:
                    print >>info_file, '%d %d' % tuple(mt.max_locs[channel_idx, max_idx]),
                print >>info_file, filename

            if not (do_maxes or do_deconv or do_deconv_norm or do_backprop or do_backprop_norm):
                continue


            with WithTimer('Load image', quiet = not do_print):
                im = caffe.io.load_image(os.path.join(datadir, filename))
            with WithTimer('Predict   ', quiet = not do_print):
                net.predict([im], oversample = False)   # Just take center crop, same as in scan_images_for_maxes

            if len(net.blobs[layer].data.shape) == 4:
                reproduced_val = net.blobs[layer].data[0,channel_idx,ii,jj]
            else:
                reproduced_val = net.blobs[layer].data[0,channel_idx]
            if abs(reproduced_val - recorded_val) > .1:
                print 'Warning: recorded value %s is suspiciously different from reproduced value %s. Is the filelist the same?' % (recorded_val, reproduced_val)

            if do_maxes:
                #grab image from data layer, not from im (to ensure preprocessing / center crop details match between image and deconv/backprop)
                out_arr = np.zeros((3,size_ii,size_jj), dtype='float32')
                out_arr[:, out_ii_start:out_ii_end, out_jj_start:out_jj_end] = net.blobs['data'].data[0,:,data_ii_start:data_ii_end,data_jj_start:data_jj_end]
                with WithTimer('Save img  ', quiet = not do_print):
                    save_caffe_image(out_arr, os.path.join(unit_dir, 'maxim_%03d.png' % max_idx_0),
                                     autoscale = False, autoscale_center = 0)
                
            if do_deconv or do_deconv_norm:
                diffs = net.blobs[layer].diff * 0
                if len(diffs.shape) == 4:
                    diffs[0,channel_idx,ii,jj] = 1.0
                else:
                    diffs[0,channel_idx] = 1.0
                with WithTimer('Deconv    ', quiet = not do_print):
                    net.deconv_from_layer(layer, diffs)

                out_arr = np.zeros((3,size_ii,size_jj), dtype='float32')
                out_arr[:, out_ii_start:out_ii_end, out_jj_start:out_jj_end] = net.blobs['data'].diff[0,:,data_ii_start:data_ii_end,data_jj_start:data_jj_end]
                if out_arr.max() == 0:
                    print 'Warning: Deconv out_arr in range', out_arr.min(), 'to', out_arr.max(), 'ensure force_backward: true in prototxt'
                if do_deconv:
                    with WithTimer('Save img  ', quiet = not do_print):
                        save_caffe_image(out_arr, os.path.join(unit_dir, 'deconv_%03d.png' % max_idx_0),
                                         autoscale = False, autoscale_center = 0)
                if do_deconv_norm:
                    out_arr = np.linalg.norm(out_arr, axis=0)
                    with WithTimer('Save img  ', quiet = not do_print):
                        save_caffe_image(out_arr, os.path.join(unit_dir, 'deconvnorm_%03d.png' % max_idx_0))

            if do_backprop or do_backprop_norm:
                diffs = net.blobs[layer].diff * 0
                diffs[0,channel_idx,ii,jj] = 1.0
                with WithTimer('Backward  ', quiet = not do_print):
                    net.backward_from_layer(layer, diffs)

                out_arr = np.zeros((3,size_ii,size_jj), dtype='float32')
                out_arr[:, out_ii_start:out_ii_end, out_jj_start:out_jj_end] = net.blobs['data'].diff[0,:,data_ii_start:data_ii_end,data_jj_start:data_jj_end]
                if out_arr.max() == 0:
                    print 'Warning: Deconv out_arr in range', out_arr.min(), 'to', out_arr.max(), 'ensure force_backward: true in prototxt'
                if do_backprop:
                    with WithTimer('Save img  ', quiet = not do_print):
                        save_caffe_image(out_arr, os.path.join(unit_dir, 'backprop_%03d.png' % max_idx_0),
                                         autoscale = False, autoscale_center = 0)
                if do_backprop_norm:
                    out_arr = np.linalg.norm(out_arr, axis=0)
                    with WithTimer('Save img  ', quiet = not do_print):
                        save_caffe_image(out_arr, os.path.join(unit_dir, 'backpropnorm_%03d.png' % max_idx_0))
                
        if do_info:
            info_file.close()
