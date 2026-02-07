# medical_analyzer
An experimental medical analyzer that uses computer vision to try to identify abnormalities in an MRI scans.

This is not a medically approved tool and is not yet accurate anough for any form of serious medical use.

The setup file installs all of the necessary vision models.

To run the setup script on most linux distros run the following:

python -m venv env

source env/bin/activate

python ./setup_advanced.py


==========================================================================

Once you have setup up all of the models you only need to install one more pip library(cv2 - opencv-python)to start using the medical_analyzer

To install it simply run the following command on same directory:

pip install opencv-python

==========================================================================

Now that you have installed all of the dependencies you can start using the medical_analyzer

To use it first paste an MRI image you want to analyze in the same directory as the medical_analyzer.py

Then use the following structure to construct your prompt:

medical_analyzer.py [-h] [--sensitivity SENSITIVITY] [--no-save] images [images ...]
