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
    parser = argparse.ArgumentParser(description='Script to find, with or without regularization, images that cause high or low activations of specific neurons in a network via numerical optimization. Settings are read from settings.py, overridden in settings_local.py, and may be further overridden on the command line.',
                                     formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100)
    )

    # Network and data options
    parser.add_argument('--caffe-root', type = str, default = settings.caffevis_caffe_root,
                        help = 'Path to caffe root directory.')
    parser.add_argument('--deploy-proto', type = str, default = settings.caffevis_deploy_prototxt,
                        help = 'Path to caffe network prototxt.')
    parser.add_argument('--net-weights', type = str, default = settings.caffevis_network_weights,
                        help = 'Path to caffe network weights.')
    parser.add_argument('--mean', type = str, default = repr(settings.caffevis_data_mean),
                        help = '''Mean. The mean may be None, a tuple of one mean value per channel, or a string specifying the path to a mean image to load. Because of the multiple datatypes supported, this argument must be specified as a string that evaluates to a valid Python object. For example: "None", "(10,20,30)", and "'mean.npy'" are all valid values. Note that to specify a string path to a mean file, it must be passed with quotes, which usually entails passing it with double quotes in the shell! Alternately, just provide the mean in settings_local.py.''')
    parser.add_argument('--channel-swap-to-rgb', type = str, default = '(2,1,0)',
                        help = 'Permutation to apply to channels to change to RGB space for plotting. Hint: (0,1,2) if your network is trained for RGB, (2,1,0) if it is trained for BGR.')
    parser.add_argument('--data-size', type = str, default = '(227,227)',
                        help = 'Size of network input.')

    #### FindParams

    # Where to start
    parser.add_argument('--start-at', type = str, default = 'mean_plus_rand', choices = ('mean_plus_rand', 'randu', 'mean'),
                        help = 'How to generate x0, the initial point used in optimization.')
    parser.add_argument('--rand-seed', type = int, default = 0,
                        help = 'Random seed used for generating the start-at image (use different seeds to generate different images).')

    # What to optimize
    parser.add_argument('--push-layer', type = str, default = 'fc8',
                        help = 'Name of layer that contains the desired neuron whose value is optimized.')
    parser.add_argument('--push-channel', type = int, default = '130',
                        help = 'Channel number for desired neuron whose value is optimized (channel for conv, neuron index for FC).')
    parser.add_argument('--push-spatial', type = str, default = 'None',
                        help = 'Which spatial location to push for conv layers. For FC layers, set this to None. For conv layers, set it to a tuple, e.g. when using `--push-layer conv5` on AlexNet, --push-spatial (6,6) will maximize the center unit of the 13x13 spatial grid.')
    parser.add_argument('--push-dir', type = float, default = 1,
                        help = 'Which direction to push the activation of the selected neuron, that is, the value used to begin backprop. For example, use 1 to maximize the selected neuron activation and  -1 to minimize it.')

    # Use regularization?
    parser.add_argument('--decay', type = float, default = 0,
                        help = 'Amount of L2 decay to use.')
    parser.add_argument('--blur-radius', type = float, default = 0,
                        help = 'Radius in pixels of blur to apply after each BLUR_EVERY steps. If 0, perform no blurring. Blur sizes between 0 and 0.3 work poorly.')
    parser.add_argument('--blur-every', type = int, default = 0,
                        help = 'Blur every BLUR_EVERY steps. If 0, perform no blurring.')
    parser.add_argument('--small-val-percentile', type = float, default = 0,
                        help = 'Induce sparsity by setting pixels with absolute value under SMALL_VAL_PERCENTILE percentile to 0. Not discussed in paper. 0 to disable.')
    parser.add_argument('--small-norm-percentile', type = float, default = 0,
                        help = 'Induce sparsity by setting pixels with norm under SMALL_NORM_PERCENTILE percentile to 0. \\theta_{n_pct} from the paper. 0 to disable.')
    parser.add_argument('--px-benefit-percentile', type = float, default = 0,
                        help = 'Induce sparsity by setting pixels with contribution under PX_BENEFIT_PERCENTILE percentile to 0. Mentioned briefly in paper but not used. 0 to disable.')
    parser.add_argument('--px-abs-benefit-percentile', type = float, default = 0,
                        help = 'Induce sparsity by setting pixels with contribution under PX_BENEFIT_PERCENTILE percentile to 0. \\theta_{c_pct} from the paper. 0 to disable.')

    # How much to optimize
    parser.add_argument('--lr-policy', type = str, default = 'constant', choices = LR_POLICY_CHOICES,
                        help = 'Learning rate policy. See description in lr-params.')
    parser.add_argument('--lr-params', type = str, default = '{"lr": 1}',
                        help = 'Learning rate params, specified as a string that evalutes to a Python dict. Params that must be provided dependon which lr-policy is selected. The "constant" policy requires the "lr" key and uses the constant given learning rate. The "progress" policy requires the "max_lr" and "desired_prog" keys and scales the learning rate such that the objective function will change by an amount equal to DESIRED_PROG under a linear objective assumption, except the LR is limited to MAX_LR. The "progress01" policy requires the "max_lr", "early_prog", and "late_prog_mult" keys and is tuned for optimizing neurons with outputs in the [0,1] range, e.g. neurons on a softmax layer. Under this policy optimization slows down as the output approaches 1 (see code for details).')
    parser.add_argument('--max-iter', type = int, default = 500,
                        help = 'Number of iterations of the optimization loop.')

    # Where to save results
    parser.add_argument('--output-prefix', type = str, default = 'optimize_results/opt',
                        help = 'Output path and filename prefix (default: optimize_results/opt)')
    parser.add_argument('--output-template', type = str, default = '%(p.push_layer)s_%(p.push_channel)04d_%(p.rand_seed)d',
                        help = 'Output filename template; see code for details (default: "%%(p.push_layer)s_%%(p.push_channel)04d_%%(p.rand_seed)d"). '
                        'The default output-prefix and output-template produce filenames like "optimize_results/opt_prob_0278_0_best_X.jpg"')
    parser.add_argument('--brave', action = 'store_true', help = 'Allow overwriting existing results files. Default: off, i.e. cowardly refuse to overwrite existing files.')
    parser.add_argument('--skipbig', action = 'store_true', help = 'Skip outputting large *info_big.pkl files (contains pickled version of x0, last x, best x, first x that attained max on the specified layer.')

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
    data_mean = eval(args.mean)

    if isinstance(data_mean, basestring):
        # If the mean is given as a filename, load the file
        try:
            data_mean = np.load(data_mean)
        except IOError:
            print '\n\nCound not load mean file:', data_mean
            print 'To fetch a default model and mean file, use:\n'
            print '  $ cd models/caffenet-yos/'
            print '  $ cp ./fetch.sh\n\n'
            print 'Or to use your own mean, change caffevis_data_mean in settings_local.py or override by running with `--mean MEAN_FILE` (see --help).\n'
            raise
        # Crop center region (e.g. 227x227) if mean is larger (e.g. 256x256)
        excess_h = data_mean.shape[1] - data_size[0]
        excess_w = data_mean.shape[2] - data_size[1]
        assert excess_h >= 0 and excess_w >= 0, 'mean should be at least as large as %s' % repr(data_size)
        data_mean = data_mean[:, (excess_h/2):(excess_h/2+data_size[0]), (excess_w/2):(excess_w/2+data_size[1])]
    elif data_mean is None:
        pass
    else:
        # The mean has been given as a value or a tuple of values
        data_mean = np.array(data_mean)
        # Promote to shape C,1,1
        while len(data_mean.shape) < 3:
            data_mean = np.expand_dims(data_mean, -1)

    print 'Using mean:', repr(data_mean)
            
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
