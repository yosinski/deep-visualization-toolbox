# Computing per-unit visualizations for a new network

Per-unit visualizations are included for the caffenet-yos network but not for other networks (the total size is at least several GB, which becomes cumbersome to distribute).

But the per-unit visualizations can be computed for any network:

* To find synthetic images that cause high activation via regularized optimization, use the [optimize_image.py](/optimize_image.py) script. Script usage is [explained here](/doc/running-optimize-image.md).

* To find images (for FC layers) or crops (for conv layers) from a set of images (e.g. the ImageNet training or validation set) that cause highest activation, use the [find_max_acts.py](/find_maxes/find_max_acts.py) script to go through the set of images and note the top K images/crops and then [crop_max_patches.py](/find_maxes/crop_max_patches.py) to use the noted max images / max locations to output the crops and/or deconv of the crops.

