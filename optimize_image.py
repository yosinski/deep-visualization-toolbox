#! /usr/bin/env python

import os
import sys
import argparse
import numpy as np

import settings
from optimize.gradient_optimizer import GradientOptimizer, FindParams
from caffevis.caffevis_helper import check_force_backward_true, read_label_file

LR_POLICY_CHOICES = ('constant', 'progress', 'progress01')



def get_parser():
    parser = argparse.ArgumentParser(description='Finds images that activate a network in various ways.')

    # Network and data options
    parser.add_argument('--caffe_root', type = str, default = settings.caffevis_caffe_root)
    parser.add_argument('--deploy_proto', type = str, default = settings.caffevis_deploy_prototxt)
    parser.add_argument('--net_weights', type = str, default = settings.caffevis_network_weights)
    parser.add_argument('--mean', type = str, default = settings.caffevis_data_mean)
    parser.add_argument('--channel_swap_to_rgb', type = str, default = '(2,1,0)', help = 'Permutation to apply to channels to change to RGB space for plotting (Hint: (0,1,2) if your network is trained for RGB, (2,1,0) if it is trained fro BGR)')
    parser.add_argument('--data_size', type = str, default = '(227,227)')

    # FindParams
    parser.add_argument('--start_at', type = str, default = 'mean_plus', choices = ('mean_plus', 'randu', 'zero'))
    parser.add_argument('--rand_seed', type = int, default = 0)
    parser.add_argument('--push_layer', type = str, default = 'prob', help = 'Which layer to push')
    parser.add_argument('--push_channel', type = int, default = '278', help = 'Which channel to push')
    parser.add_argument('--push_spatial', type = str, default = 'None', help = 'Which spatial location to push for conv layers. For FC layers, set this to None. For conv layers, set it to a tuple, e.g. when using `--push_layer conv5` on AlexNet, --push_spatial (6,6) will maximize the center unit of the 13x13 spatial grid.')
    parser.add_argument('--push_dir', type = float, default = 1.0, help = 'Push direction, or value to initialize backprop with. Positive to maximize, negative to minimize.')
    parser.add_argument('--decay', type = float, default = .01)
    parser.add_argument('--blur_radius', type = float, default = 0)
    parser.add_argument('--blur_every', type = int, default = 0)
    parser.add_argument('--small_val_percentile', type = float, default = 0)
    parser.add_argument('--small_norm_percentile', type = float, default = 0)
    parser.add_argument('--px_benefit_percentile', type = float, default = 0)
    parser.add_argument('--px_abs_benefit_percentile', type = float, default = 0)
    parser.add_argument('--lr_policy', type = str, default = 'constant', choices = LR_POLICY_CHOICES)
    parser.add_argument('--lr_params', type = str, default = '{"lr": .01}')
    parser.add_argument('--max_iter', type = int, default = 300)

    # Results output options
    parser.add_argument('--output_prefix', type = str, default = 'optimize_results/opt')
    parser.add_argument('--output_template', type = str, default = '%(p.push_layer)s_%(p.push_channel)04d_%(p.rand_seed)d')
    parser.add_argument('--brave', action = 'store_true', help = 'Allow overwriting existing results files. Default: cowardly refuse to overwrite, exiting instead.')
    parser.add_argument('--skipbig', action = 'store_true', help = 'Skip outputting large *info_big.pkl files (contains pickled version of x0, last x, best x, first x that attained max on the specified layer')

    return parser



def parse_and_validate_lr_params(parser, lr_policy, lr_params):
    assert lr_policy in LR_POLICY_CHOICES

    try:
        lr_params = eval(lr_params)
    except (SyntaxError,NameError) as _:
        err = 'Tried to parse the following lr_params value\n%s\nas a Python expression, but it failed. lr_params should evaluate to a valid Python dict.' % lr_params
        parser.error(err)

    if lr_policy == 'constant':
        if not 'lr' in lr_params:
            parser.error('Expected lr_params to be dict with at least "lr" key, but dict is %s' % repr(lr_params))
    elif lr_policy == 'progress':
        if not ('max_lr' in lr_params and 'desired_prog' in lr_params):
            parser.error('Expected lr_params to be dict with at least "max_lr" and "desired_prog" keys, but dict is %s' % repr(lr_params))
    elif lr_policy == 'progress01':
        if not ('max_lr' in lr_params and 'early_prog' in lr_params and 'late_prog_mult' in lr_params):
            parser.error('Expected lr_params to be dict with at least "max_lr", "early_prog", and "late_prog_mult" keys, but dict is %s' % repr(lr_params))

    return lr_params



def parse_and_validate_push_spatial(parser, push_spatial):
    '''Returns tuple of length 2.'''
    try:
        push_spatial = eval(push_spatial)
    except (SyntaxError,NameError) as _:
        err = 'Tried to parse the following push_spatial value\n%s\nas a Python expression, but it failed. push_spatial should be a valid Python expression.' % push_spatial
        parser.error(err)

    if push_spatial == None:
        push_spatial = (0,0)    # Convert to tuple format
    elif isinstance(push_spatial, tuple) and len(push_spatial) == 2:
        pass
    else:
        err = 'push_spatial should be None or a valid tuple of indices of length 2, but it is: %s' % push_spatial
        parser.error(err)

    return push_spatial



def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # Finish parsing args
    channel_swap_to_rgb = eval(args.channel_swap_to_rgb)
    assert isinstance(channel_swap_to_rgb, tuple) and len(channel_swap_to_rgb) > 0, 'channel_swap_to_rgb should be a tuple'
    data_size = eval(args.data_size)
    assert isinstance(data_size, tuple) and len(data_size) == 2, 'data_size should be a length 2 tuple'
    #channel_swap_inv = tuple([net_channel_swap.index(ii) for ii in range(len(net_channel_swap))])

    lr_params = parse_and_validate_lr_params(parser, args.lr_policy, args.lr_params)
    push_spatial = parse_and_validate_push_spatial(parser, args.push_spatial)
    
    # Load mean
    try:
        data_mean = np.load(args.mean)
    except IOError:
        print '\n\nCound not load mean file:', args.mean
        print 'To fetch a default model and mean file, use:\n'
        print '  $ cd models/caffenet-yos/'
        print '  $ ./fetch.sh\n\n'
        print 'Or to use your own mean, change caffevis_data_mean in settings.py or override by running with `--mean MEAN_FILE`.\n'
        raise
    # Crop center region (e.g. 227x227) if mean is larger (e.g. 256x256)
    excess_h = data_mean.shape[1] - data_size[0]
    excess_w = data_mean.shape[2] - data_size[1]
    assert excess_h >= 0 and excess_w >= 0, 'mean should be at least as large as %s' % repr(data_size)
    data_mean = data_mean[:, (excess_h/2):(excess_h/2+data_size[0]), (excess_w/2):(excess_w/2+data_size[1])]
    
    # Load network
    sys.path.insert(0, os.path.join(args.caffe_root, 'python'))
    import caffe
    net = caffe.Classifier(
        args.deploy_proto,
        args.net_weights,
        mean = data_mean,
        raw_scale = 1.0,
    )
    check_force_backward_true(settings.caffevis_deploy_prototxt)

    labels = None
    if settings.caffevis_labels:
        labels = read_label_file(settings.caffevis_labels)

    optimizer = GradientOptimizer(net, data_mean, labels = labels,
                                  label_layers = settings.caffevis_label_layers,
                                  channel_swap_to_rgb = channel_swap_to_rgb)
    
    params = FindParams(
        start_at = args.start_at,
        rand_seed = args.rand_seed,
        push_layer = args.push_layer,
        push_channel = args.push_channel,
        push_spatial = push_spatial,
        push_dir = args.push_dir,
        decay = args.decay,
        blur_radius = args.blur_radius,
        blur_every = args.blur_every,
        small_val_percentile = args.small_val_percentile,
        small_norm_percentile = args.small_norm_percentile,
        px_benefit_percentile = args.px_benefit_percentile,
        px_abs_benefit_percentile = args.px_abs_benefit_percentile,
        lr_policy = args.lr_policy,
        lr_params = lr_params,
        max_iter = args.max_iter,
    )

    prefix_template = '%s_%s_' % (args.output_prefix, args.output_template)
    im = optimizer.run_optimize(params, prefix_template = prefix_template,
                                brave = args.brave, skipbig = args.skipbig)



if __name__ == '__main__':
    main()
