# Deep Visualization Toolbox Installation



### Step 0: Compile master branch of caffe (optional)

Get the master branch of [Caffe](http://caffe.berkeleyvision.org/) to compile on your
machine. If you've never used Caffe before, it can take a bit of time to get all the required libraries in place. Fortunately, the [installation process is well documented](http://caffe.berkeleyvision.org/installation.html).

Note: You can set `CPU_ONLY := 1` in your `Makefile.config` to skip all the Cuda/GPU stuff. The Deep Visualization Toolbox can run with Caffe in either CPU or GPU mode.



### Step 1: Compile deconv-deep-vis-toolbox branch of caffe

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



### Step 2: Download and configure Deep Visualization Toolbox code

You can put it wherever you like:

    $ git clone https://github.com/yosinski/deep-visualization-toolbox
    $ cd deep_visualization_toolbox

Copy settings.py.template to settings.py and edit it so the `caffevis_caffe_root` variable points to the directory where you've compiled caffe in Step 1:

    $ cp settings.py.template settings.py
    $ < edit settings.py >

Download the example model weights and corresponding top-9 visualizations saved as jpg (downloads a 230MB model and 1.1GB of jpgs to show as visualization):

    $ cd models/caffenet-yos/
    $ ./fetch.sh



### Step 3: Run it!

Simple:

    $ ./run_toolbox.py

Once the toolbox is running, push 'h' to show a help screen. You can also have a look at bindings.py to see what the various keys do.




## Troubleshooting

If you have any problems getting the code running, please feel free to [email me](http://yosinski.com/). I might have left out an important detail here or there :).
