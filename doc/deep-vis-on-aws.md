# Using the deep-vis toolbox on an Amazon AWS EC2 instance

Building Caffe from scrach on your local machine can be a pain. These steps will help get the deep-vis toolbox set up and displaying on an AWS EC2 instance built from the Caffe/CUDA/CuDNN AMI. Because the EC2 instance is a remote server, the deep-vis toolbox will not be able to display to your desktop directly-- these steps allow you to forward the remote desktop to your local screen using a VNC viewer. 

## Setting up the AMI and VNC viewer

1. First off, grab the AMI that the folks at Caffe built (ami-763a311e) and get it up and running on a g2.2xlarge or g2.8xlarge instance. Edit your inbound security permissions to allow all traffic from your IP address. 

2. Clone this repo to your AMI and follow the basic instructions in the README to get the deep vis toolbox set up with the necessary dependencies.

3. You'll need to set up a new user on the instance in order to forward the display to your local computer. Follow the instructions [here](http://www.brianlinkletter.com/how-to-set-up-a-new-userid-on-your-amazon-aws-server-instance/). Next, you'll need to set up the VNC server, for which there are good instructions [here](http://www.brianlinkletter.com/how-to-run-gui-applications-xfce-on-an-amazon-aws-cloud-server-instance/). If you're on a Mac, I'd recommend using [Chicken of the VNC](http://sourceforge.net/projects/cotvnc/). 

4. After you've logged in using SSH port forwarding, start the vncserver, and note what number desktop it is located at, ie. 'ip-172-31-48-75:**1**'. In Chicken of the VNC, open a new connection. The host is your instance's public DNS, the display is the # specified by VNC, and the password is the one you set earlier. 

5. Once you're logged in, you'll be able to move around the terminal on the remote desktop to deep-visualization-toolbox and run `run_toolbox.py` without any display issues!
