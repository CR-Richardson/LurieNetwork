# Lurie Network
This code acompanies the papers [Lurie Networks with Robust k-contracting Dynamics](https://openreview.net/forum?id=3Jm4dbrKGZ) and [Lurie Networks with k-contracting Dynamics](https://openreview.net/forum?id=RaAYeCxj1u). The repository contains code for generating the synthetic dynamical systems data, implementations of the models, and code for training the models on the synthetic dynamical systems examples and FMNIST. 

### Authors:
* Carl R Richardson (cr2g16@soton.ac.uk)
* Matthew C Turner (m.c.turner@soton.ac.uk)
* Steve R Gunn (srg@ecs.soton.ac.uk)

## Prerequisites
All the code is written in Python and predominantly in PyTorch. This must be installed along with several other standard libraries such as Numpy, Matplotlib, etc. For the Neural ODE, torchdiffeq must also be installed. For easy use, please use the same file structure as below.

## Overview
The repository is organised as follows:
* `Code`
  * `DataGeneration`
    *  `DataGeneration.py` : Script for generating data for the dynamical systems examples.
  * `Models`
    * A set of Python files containing the class of each model implemented in the papers.
  * `Train`
          * `Train_FMNIST.py` : Script for training the models on FMNIST.
	  * `Train.py` : Script for training the models on the dynamical systems datasets.
	  * `Train_Tools.py` : Back-end code for `Train.py`.
  * `Utils`
          * Plots_Tools.py : A set of functions used to create the different plots seen in the papers.

 * `Data` : Directory for storing datasets.
 * `Models` : Directory for storing models and associated files.

## Getting Started
- Download dependencies.
- Add root directory to relevant scripts.
- Set hyperparameters, specify model and dataset, then run either `Train.py` or `Train_FMNIST.py`.
