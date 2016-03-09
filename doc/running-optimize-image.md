# Producing optimized images

## Reproducing paper results

The optimization results in the paper may be reproduced using the `optimize_image.py` script. Table 1 of the paper gives four sets of recommended regularization hyperparameters. These may be implemented using the following combination of command line flags (in the same order as Table 1):

    ./optimize_image.py --decay 0      --blur-radius 0.5 --blur-every 4  --small-norm-percentile 50     --max-iter 500  --lr-policy progress --lr-params "{'max_lr': 100.0, 'desired_prog': 2.0}"
    ./optimize_image.py --decay 0.3    --blur-radius 0   --blur-every 0  --small-norm-percentile 20     --max-iter 750  --lr-policy constant --lr-params "{'lr': 100.0}"
    ./optimize_image.py --decay 0.0001 --blur-radius 1.0 --blur-every 4                                 --max-iter 1000 --lr-policy constant --lr-params "{'lr': 100.0}"
    ./optimize_image.py --decay 0      --blur-radius 0.5 --blur-every 4  --px-abs-benefit-percentile 90 --max-iter 1000 --lr-policy progress --lr-params "{'max_lr': 100000000, 'desired_prog': 2.0}"

The hyperparameters given on the third line are the ones used for most of the image in the paper. This set of hyperparameters produced smooth images by using a wide 1 pixel radius blur applied every 4 steps. See below for a complete description of other available `optimize_image.py` options.


## Example optimization output

Using a fresh checkout with the default model downloaded, we can generate an image of a flamingo using the following command:

    ./optimize_image.py --push-layer fc8 --push-channel 130                  \
        --decay 0.0001 --blur-radius 1.0 --blur-every 4                      \
        --max-iter 1000 --lr-policy constant --lr-params "{'lr': 100.0}"
        
This produces a few files (with comments added):

    [deep-visualization-toolbox] $ cd optimize_results/
    [deep-visualization-toolbox/optimize_results] $ ls
    opt_fc8_0130_0_best_X.jpg           # resulting image
    opt_fc8_0130_0_best_Xpm.jpg         # resulting image plus mean image
    opt_fc8_0130_0_info.txt             # text description of optimization parameters and results
    opt_fc8_0130_0_info.pkl             # pickle file containing all results (except images)
    opt_fc8_0130_0_info_big.pkl         # pickle file containing all results

The two jpg images that are output (without and with mean added) look like this:

![Best X found](/doc/opt_fc8_0130_0_best_X.jpg?raw=true "Best X found")
![Best X found plus mean](/doc/opt_fc8_0130_0_best_Xpm.jpg?raw=true "Best X found plus mean")

We can examine the `opt_fc8_0130_0_info.txt` file to see a record of the hyperparameters that were used and the results of the optimization:

    [deep-visualization-toolbox/optimize_results] $ cat opt_fc8_0130_0_info.txt
    FindParams:
                     blur_every: 4
                    blur_radius: 1.0
                          decay: 0.0001
                      lr_params: {'lr': 100.0}
                      lr_policy: constant
                       max_iter: 1000
                   push_channel: 130
                       push_dir: 1
                     push_layer: fc8
                   push_spatial: (0, 0)
                      push_unit: (130, 0, 0)
      px_abs_benefit_percentile: 0
          px_benefit_percentile: 0
                      rand_seed: 0
          small_norm_percentile: 0
           small_val_percentile: 0
                       start_at: mean_plus_rand
    
    
    FindResults:
                        best_ii: 992
                       best_obj: 71.1864
                        best_xx: (3, 227, 227) array [0.505416410453, 0.499939193577, ...]
                           dist: [1.9916364387e-12, 3413.05052366, ..., 3956.88723303, 3957.78892399]
                         idxmax: [(330, 0, 0), (533, 0, 0), ..., (130, 0, 0), (130, 0, 0)]
                             ii: [0, 1, ..., 998, 999]
                          ismax: [False, False, ..., True, True]
                        last_ii: 999
                       last_obj: 68.1828
                        last_xx: (3, 227, 227) array [0.506889683633, 0.501422513477, ...]
                    majority_ii: 27
                   majority_obj: 8.44836
                    majority_xx: (3, 227, 227) array [3.81062396723, 3.28850835202, ...]
                    meta_result: Metaresult: majority success
                           norm: [3921.52364497, 1118.42893441, ..., 639.989254329, 647.378869524]
                            obj: [-1.30543, -0.756828, ..., 63.5805, 68.1828]
                            std: [9.97384688005, 2.84414857575, ..., 1.62728852151, 1.6460861413]
                             x0: (3, 227, 227) array [17.6405234597, 4.00157208367, ...]

The text file output is provided just for convenience. To process the fields programmatically in Python, it's easiest to load and inspect the associated `opt_fc8_0130_0_info_big.pkl` file using the `pickle` module:

    [deep-visualization-toolbox/optimize_results] $ cd ..
    [deep-visualization-toolbox] $ python
    >>> import pickle
    >>> with open('optimize_results/opt_fc8_0130_0_info_big.pkl') as ff:
    ...     results = pickle.load(ff)
    ...
    >>> print results
    (<optimize.gradient_optimizer.FindParams object at 0x105223a10>, <optimize.gradient_optimizer.FindResults object at 0x132c29550>)
    >>> find_params, find_results = results
    >>> print find_params
    FindParams:
                        blur_every: 4
                       blur_radius: 1.0
                             decay: 0.0001
                         lr_params: {'lr': 100.0}
                         lr_policy: constant
                          max_iter: 1000
                      push_channel: 130
                          push_dir: 1
                        push_layer: fc8
                      push_spatial: (0, 0)
                         push_unit: (130, 0, 0)
         px_abs_benefit_percentile: 0
             px_benefit_percentile: 0
                         rand_seed: 0
             small_norm_percentile: 0
              small_val_percentile: 0
                          start_at: mean_plus_rand
    
    >>> print find_results
    FindResults:
                           best_ii: 992
                          best_obj: 71.1864
                           best_xx: (3, 227, 227) array [0.505416410453, 0.499939193577, ...]
                              dist: [1.9916364387e-12, 3413.05052366, ..., 3956.88723303, 3957.78892399]
                            idxmax: [(330, 0, 0), (533, 0, 0), ..., (130, 0, 0), (130, 0, 0)]
                                ii: [0, 1, ..., 998, 999]
                             ismax: [False, False, ..., True, True]
                           last_ii: 999
                          last_obj: 68.1828
                           last_xx: (3, 227, 227) array [0.506889683633, 0.501422513477, ...]
                       majority_ii: 27
                      majority_obj: 8.44836
                       majority_xx: (3, 227, 227) array [3.81062396723, 3.28850835202, ...]
                       meta_result: Metaresult: majority success
                              norm: [3921.52364497, 1118.42893441, ..., 639.989254329, 647.378869524]
                               obj: [-1.30543, -0.756828, ..., 63.5805, 68.1828]
                               std: [9.97384688005, 2.84414857575, ..., 1.62728852151, 1.6460861413]
                                x0: (3, 227, 227) array [17.6405234597, 4.00157208367, ...]
    
    >>> find_results.x0.shape
    (3, 227, 227)
    >>> find_results.last_xx.shape
    (3, 227, 227)
    >>> find_results.best_obj
    71.186447
    >>> max(find_results.obj)
    71.186447
    ...



## Full `optimize_image.py` script options

Below is a description of all available options in `optimize_image.py` script (annotated output of `./optimize_image.py --help`):

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
      
Which trained network to load

      --caffe-root CAFFE_ROOT
                            Path to caffe root directory. (default: <read from settings>)
      --deploy-proto DEPLOY_PROTO
                            Path to caffe network prototxt. (default: <read from settings>)
      --net-weights NET_WEIGHTS
                            Path to caffe network weights. (default: <read from settings>)
      --mean MEAN           Mean. The mean may be None, a tuple of one mean value per channel, or a
                            string specifying the path to a mean image to load. Because of the multiple
                            datatypes supported, this argument must be specified as a string that
                            evaluates to a valid Python object. For example: "None", "(10,20,30)", and
                            "'mean.npy'" are all valid values. Note that to specify a string path to a
                            mean file, it must be passed with quotes, which usually entails passing it
                            with double quotes in the shell! Alternately, just provide the mean in
                            settings_local.py. (default: <read from settings>)
      --channel-swap-to-rgb CHANNEL_SWAP_TO_RGB
                            Permutation to apply to channels to change to RGB space for plotting. Hint:
                            (0,1,2) if your network is trained for RGB, (2,1,0) if it is trained for
                            BGR. (default: (2,1,0))
      --data-size DATA_SIZE
                            Size of network input. (default: (227,227))
                            
Where to start optimization

      --start-at {mean_plus_rand,randu,mean}
                            How to generate x0, the initial point used in optimization. (default:
                            mean_plus_rand)
      --rand-seed RAND_SEED
                            Random seed used for generating the start-at image (use different seeds to
                            generate different images). (default: 0)

Which neuron to optimize and in what direction:

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

Which regularization options to apply

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

What learning rate schedule to use

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
      
Where to save results

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
