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




















