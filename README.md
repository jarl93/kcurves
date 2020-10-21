# k-curves
This project is an attempt to use curves (functions) and deep auto-encoders to do clustering.

# Installation
1. Please install the libraries needed listed in the requirements.txt file.
2. Clone the project: git clone https://github.com/hci-unihd/k-curves.git

# Generate synthetic data
In order to generate synthetic data run the script generate_synthetic.py with the following command:

python3 kcurves/generate_synthetic.py ./configs/synthetic_generation.yaml

The default configuration file "./configs/synthetic_generation.yaml" is a configuration file to set the functions parameters to generate the data. You can create your own file or modify default one, if you create a new one please change the last parameter when executing the script.

# Visualize the data generated

In order to visualize the data generated you need to do run tensorboard as follows:

tensorboard --logdir ./data/synthetic/plots

Then you have to go to localhost in the indicated port after running the above command.

# Train
In order to make the training run the main script in mode train by executing the following command:

python3 kcurves train ./configs/synthetic_identity_both.yaml

In this case the default configuration file is "./configs/synthetic_identity_both.yaml" but you can create a new one or modify this one.

# Test

In order to make the testing run the main script in mode test by executing the following command:

python3 kcurves test ./configs/synthetic_identity_both.yaml  

In this case the default configuration file is "./configs/synthetic_identity_both.yaml" but you can create a new one or modify this one.


# Visualize the encoder output afeter training and testing

In order to visualize the encoder output you need to run tensorboard as follows:

tensorboard --logdir ./models/synthetic

Then you have to go to localhost in the indicated port after running the above command.









