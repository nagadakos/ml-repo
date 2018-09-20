# ml-repo
A repository for understanding and developing Machine Learning Algorithms!



]>  KERAS
----------------------------------------------------------------

Envirnment used:

					Backend: 	tensorflow-gpu 1.10.1
					Interface: 	Keras 2.2.2			
					Deployment: FLASK
					Python:     V3.6


Replication instructions:

A python virtual environment was used to avoid conflict with existing isntallations.
To set this up, you can do the following.

1. Download virtual env

pip3.6 install --upgrade virtualenv

2. Create a virtual env as the root folder of the project

virtualend kerasenv

3. Activate the Virtual env (at the root folder)

source kerasenv/bin/activate

4. Install required software according to operating system.
   	Keep in mind that latest tensorflow-gpu releases require CUDA 9+,
	which in turn requires compute capability above 3.
 
pip3 tensorflow-gpu keras FLASK h5py


NOTE: If on an older system such as Ubuntu 14.04 with CUda 8.0, Tensorflow-gpu_1.4.1
is required. You can set this up with:

pip3 isntall tensorflow-gpu==1.4.1

]>  PyTorch
----------------------------------------------------------------

Envirnment used:

					Backend: 	Torch
					Interface: 	PyTorch v0.4			
					Deployment: -
					Python:     V3.6


Replication instructions:
There are 2 ways to install required software based on your prefared module managers
A. conda or B. pip. The following assumes you have a working CUDA 8.x installation.

A. Conda
1. conda install pytorch torchvision cuda80 -c pytorch

B. PIP + CUDA 8.x

# Python 2.7
pip install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp27-cp27mu-linux_x86_64.whl
pip install torchvision
# if the above command does not work, then you have python 2.7 UCS2, use this command
pip install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp27-cp27m-linux_x86_64.whl
# Python 3.5
pip3 install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp35-cp35m-linux_x86_64.whl
pip3 install torchvision
# Python 3.6
pip3 install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision
# Python 3.7
pip3 install http://download.pytorch.org/whl/cu80/torch-0.4.1.post2-cp37-cp37m-linux_x86_64.whl
pip3 install torchvision



# ##########################################################
#
#		Architecture Overview
#
# ##########################################################

Section 1 - Introduction.
-------------------------------

The files found in this repository are Deep Netowrk architectures designed
to perform character recognition on the famous MNIST handwritten characters data
set. Each folder from the Keras, PyTorch, Gluon list contains architectures 
described in that framework. The results are graphically presented in the plots
folder, models are saved in the models folder. At ther root level, top level 
modules are stored such as the plotter.py, a module used to draw various curves
to visualize data, accuracy, losss etc.












