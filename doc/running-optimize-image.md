# Producing optimized images

The optimization results in the paper may be reproduced using the `optimize_image.py` script. Table 1 of the paper gives four sets of recommended optimization hyperparameters. These may be implemented using the following combination of command line flags (same order as Table 1):

    ./optimize_image.py --decay 0      --blur-radius 0.5 --blur-every 4  --small-norm-percentile 50     --max-iter 500  --lr-policy progress --lr-params "{'max_lr': 100.0, 'desired_prog': 2.0}"
    ./optimize_image.py --decay 0.3    --blur-radius 0   --blur-every 0  --small-norm-percentile 20     --max-iter 750  --lr-policy constant --lr-params "{'lr': 100.0}"
    ./optimize_image.py --decay 0.0001 --blur-radius 1.0 --blur-every 4                                 --max-iter 1000 --lr-policy constant --lr-params "{'lr': 100.0}"
    ./optimize_image.py --decay 0      --blur-radius 0.5 --blur-every 4  --px-abs-benefit-percentile 90 --max-iter 1000 --lr-policy progress --lr-params "{'max_lr': 100000000, 'desired_prog': 2.0}"

The third line version is the most used in the paper (the smoothest, with a wide 1 pixel radius blur applied every 4 steps).

See below for a complete description of script options.



# `optimize_image.py` script options

Description of available options in `optimize_image.py` script (output of `./optimize_image.py --help`):

    usage: optimize_image.py [-h] [--caffe-root CAFFE_ROOT] [--deploy-proto DEPLOY_PROTO]
                             [--net-weights NET_WEIGHTS] [--mean MEAN]
                             [--channel-swap-to-rgb CHANNEL_SWAP_TO_RGB] [--data-size DATA_SIZE]
                             [--start-at {mean_plus_rand,randu,mean}] [--rand-seed RAND_SEED]
                             [--push-layer PUSH_LAYER] [--push-channel PUSH_CHANNEL]
                             [--push-spatial PUSH_SPATIAL] [--push-dir PUSH_DIR] [--decay DECAY]
                             [--blur-radius BLUR_RADIUS] [--blur-every BLUR_EVERY]
                             [--small-val-percentile SMALL_VAL_PERCENTILE]
                             [--small-norm-percentile SMALL_NORM_PERCENTILE]
                             [--px-benefit-percentile PX_BENEFIT_PERCENTILE]
                             [--px-abs-benefit-percentile PX_ABS_BENEFIT_PERCENTILE]
                             [--lr-policy {constant,progress,progress01}] [--lr-params LR_PARAMS]
                             [--max-iter MAX_ITER] [--output-prefix OUTPUT_PREFIX]
                             [--output-template OUTPUT_TEMPLATE] [--brave] [--skipbig]
    
    Script to find, with or without regularization, images that cause high or low activations of
    specific neurons in a network via numerical optimization.
    
    optional arguments:
      -h, --help            show this help message and exit
      --caffe-root CAFFE_ROOT
                            Path to caffe root directory. (default: /Users/jason/s/caffe2)
      --deploy-proto DEPLOY_PROTO
                            Path to caffe network prototxt. (default: /Users/jason/s/deep-visualization-
                            toolbox/models/caffenet-yos/caffenet-yos-deploy.prototxt)
      --net-weights NET_WEIGHTS
                            Path to caffe network weights. (default: /Users/jason/s/deep-visualization-
                            toolbox/models/caffenet-yos/caffenet-yos-weights)
      --mean MEAN           Path to mean image. (default: /Users/jason/s/deep-visualization-
                            toolbox/models/caffenet-yos/ilsvrc_2012_mean.npy)
      --channel-swap-to-rgb CHANNEL_SWAP_TO_RGB
                            Permutation to apply to channels to change to RGB space for plotting. Hint:
                            (0,1,2) if your network is trained for RGB, (2,1,0) if it is trained for
                            BGR. (default: (2,1,0))
      --data-size DATA_SIZE
                            Size of network input. (default: (227,227))
      --start-at {mean_plus_rand,randu,mean}
                            How to generate x0, the initial point used in optimization. (default:
                            mean_plus_rand)
      --rand-seed RAND_SEED
                            Random seed used for generating the start-at image (use different seeds to
                            generate different images). (default: 0)
      --push-layer PUSH_LAYER
                            Name of layer that contains the desired neuron whose value is optimized.
                            (default: fc8)
      --push-channel PUSH_CHANNEL
                            Channel number for desired neuron whose value is optimized (channel for
                            conv, neuron index for FC). (default: 130)
      --push-spatial PUSH_SPATIAL
                            Which spatial location to push for conv layers. For FC layers, set this to
                            None. For conv layers, set it to a tuple, e.g. when using `--push-layer
                            conv5` on AlexNet, --push-spatial (6,6) will maximize the center unit of the
                            13x13 spatial grid. (default: None)
      --push-dir PUSH_DIR   Which direction to push the activation of the selected neuron, that is, the
                            value used to begin backprop. For example, use 1 to maximize the selected
                            neuron activation and -1 to minimize it. (default: 1)
      --decay DECAY         Amount of L2 decay to use. (default: 0)
      --blur-radius BLUR_RADIUS
                            Radius in pixels of blur to apply after each BLUR_EVERY steps. If 0, perform
                            no blurring. Blur sizes between 0 and 0.3 work poorly. (default: 0)
      --blur-every BLUR_EVERY
                            Blur every BLUR_EVERY steps. If 0, perform no blurring. (default: 0)
      --small-val-percentile SMALL_VAL_PERCENTILE
                            Induce sparsity by setting pixels with absolute value under
                            SMALL_VAL_PERCENTILE percentile to 0. Not discussed in paper. 0 to disable.
                            (default: 0)
      --small-norm-percentile SMALL_NORM_PERCENTILE
                            Induce sparsity by setting pixels with norm under SMALL_NORM_PERCENTILE
                            percentile to 0. \theta_{n_pct} from the paper. 0 to disable. (default: 0)
      --px-benefit-percentile PX_BENEFIT_PERCENTILE
                            Induce sparsity by setting pixels with contribution under
                            PX_BENEFIT_PERCENTILE percentile to 0. Mentioned briefly in paper but not
                            used. 0 to disable. (default: 0)
      --px-abs-benefit-percentile PX_ABS_BENEFIT_PERCENTILE
                            Induce sparsity by setting pixels with contribution under
                            PX_BENEFIT_PERCENTILE percentile to 0. \theta_{c_pct} from the paper. 0 to
                            disable. (default: 0)
      --lr-policy {constant,progress,progress01}
                            Learning rate policy. See description in lr-params. (default: constant)
      --lr-params LR_PARAMS
                            Learning rate params, specified as a string that evalutes to a Python dict.
                            Params that must be provided dependon which lr-policy is selected. The
                            "constant" policy requires the "lr" key and uses the constant given learning
                            rate. The "progress" policy requires the "max_lr" and "desired_prog" keys
                            and scales the learning rate such that the objective function will change by
                            an amount equal to DESIRED_PROG under a linear objective assumption, except
                            the LR is limited to MAX_LR. The "progress01" policy requires the "max_lr",
                            "early_prog", and "late_prog_mult" keys and is tuned for optimizing neurons
                            with outputs in the [0,1] range, e.g. neurons on a softmax layer. Under this
                            policy optimization slows down as the output approaches 1 (see code for
                            details). (default: {"lr": 1})
      --max-iter MAX_ITER   Number of iterations of the optimization loop. (default: 500)
      --output-prefix OUTPUT_PREFIX
                            Output path and filename prefix (default: optimize_results/opt) (default:
                            optimize_results/opt)
      --output-template OUTPUT_TEMPLATE
                            Output filename template; see code for details (default:
                            "%(p.push_layer)s_%(p.push_channel)04d_%(p.rand_seed)d"). The default
                            output-prefix and output-template produce filenames like
                            "optimize_results/opt_prob_0278_0_best_X.jpg" (default:
                            %(p.push_layer)s_%(p.push_channel)04d_%(p.rand_seed)d)
      --brave               Allow overwriting existing results files. Default: off, i.e. cowardly refuse
                            to overwrite existing files. (default: False)
      --skipbig             Skip outputting large *info_big.pkl files (contains pickled version of x0,
                            last x, best x, first x that attained max on the specified layer. (default:
                            False)
