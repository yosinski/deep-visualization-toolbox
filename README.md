# Deep Visualization Toolbox Installation



### Step 0: Compile master branch of caffe (optional)

Get the master branch of [Caffe](http://caffe.berkeleyvision.org/) to compile on your
machine. If you've never used Caffe before, it can take a bit of time to get all the required libraries in place. Fortunately, the [installation process is well documented](http://caffe.berkeleyvision.org/installation.html). When you're installing OpenCV, install the Python bindings as well (see Step 2 below).

Note: You can set `CPU_ONLY := 1` in your `Makefile.config` to skip all the Cuda/GPU stuff. The Deep Visualization Toolbox can run with Caffe in either CPU or GPU mode.



### Step 1: Compile the deconv-deep-vis-toolbox branch of caffe

Instead of using the master branch of caffe, to use the demo
you'll need a slightly modified branch (supporting deconv and a few
extra Python bindings). Getting the branch and switching to it is easy.
Starting from your caffe directory, run:

    $ git remote add yosinski https://github.com/yosinski/caffe.git
    $ git fetch --all
    $ git checkout --track -b deconv-deep-vis-toolbox yosinski/deconv-deep-vis-toolbox
    $ make clean
    $ make -j
    $ make -j pycaffe

As noted above, feel free to compile in `CPU_ONLY` mode if desired.



### Step 2: Install python-opencv

You may have already installed the `python-opencv` bindings as part of the Caffe setup process. If `import cv2` works from Python, then you're all set. If not, install the bindings like this:

Linux:

    $ sudo apt-get install python-opencv

Mac using [homebrew](http://brew.sh/) (if desired, add option `--with-tbb` to enable parallel code using Intel TBB):

    $ brew install opencv

Other install options on Mac may also work well.



### Step 3: Download and configure Deep Visualization Toolbox code

You can put it wherever you like:

    $ git clone https://github.com/yosinski/deep-visualization-toolbox
    $ cd deep_visualization_toolbox

Copy `settings.py.template` to `settings.py` and edit it so the `caffevis_caffe_root` variable points to the directory where you've compiled caffe in Step 1:

    $ cp settings.py.template settings.py
    $ < edit settings.py >

Download the example model weights and corresponding top-9 visualizations saved as jpg (downloads a 230MB model and 1.1GB of jpgs to show as visualization):

    $ cd models/caffenet-yos/
    $ ./fetch.sh



### Step 4: Run it!

Simple:

    $ ./run_toolbox.py

Once the toolbox is running, push 'h' to show a help screen. You can also have a look at `bindings.py` to see what the various keys do. If the window is too large or too small for your screen, set the `global_scale` variable in `settings.py` to a value smaller or larger than one.




# Troubleshooting

If you have any problems running the Deep Vis Toolbox, here are a few things to try:

 * Try using the `dev` branch instead (`git checkout dev`). Sometimes it's a little more up to date than the master branch.
 * If you get an error (`AttributeError: 'Classifier' object has no attribute 'backward_from_layer'`) when switching to backprop or deconv modes, it's because your compiled branch of Caffe does not have the necessary Python bindings for backprop/deconv. Follow the directions in "Step 1: Compile the deconv-deep-vis-toolbox branch of caffe" above.
 * If the backprop pane in the lower left is just gray, it's probably because backprop and deconv are producing all zeros. By default, Caffe won't compute derivatives at the data layer, because they're not needed to update parameters. The fix is simple: just add "force_backward: true" to your network prototxt, [like this](https://github.com/yosinski/deep-visualization-toolbox/blob/master/models/caffenet-yos/caffenet-yos-deploy.prototxt#L7).
 * If none of that helps, feel free to [email me](http://yosinski.com/) or [submit an issue](https://github.com/yosinski/deep-visualization-toolbox/issues). I might have left out an important detail here or there :).
