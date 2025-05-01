# Lurie Network
This code acompanies the papers [Lurie Networks with Robust Convergent Dynamics](https://openreview.net/forum?id=3Jm4dbrKGZ) and [Lurie Networks with k-contracting Dynamics](https://openreview.net/forum?id=RaAYeCxj1u). The repository contains code for generating the synthetic dynamical systems data, implementations of the proposed models / baselines used for comparison, and training scripts for the synthetic dynamical systems and FMNIST datasets. 

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
    * `ModelComp_Tools.py` : A set of functions used to evaluate/compare different models.
    * `Plots_Tools.py` : A set of functions used to create the different plots seen in the papers.
    * `tSNE.py` : Function for creating tSNE plot.

 * `Data` : Directory for storing datasets.
 * `Models` : Directory for storing models and associated files.

## Getting Started
- Download dependencies.
- Add root directory to relevant scripts.
- For dynamical systems examples: specify dataset parameters then generate data by running `DataGeneration.py`. Specify model and set hyperparameters, then run `Train.py`.
- For FMNIST, download dataset, specify model, set hyperparameters, then run `Train_FMNIST.py`.
