# Lurie Network
This code acompanies the papers [Lurie Networks with Robust k-contracting Dynamics](https://openreview.net/forum?id=3Jm4dbrKGZ) where the experimental setup is detailed in the *Empirical Evaluation* section. The repository contains the code for generating the synthetic experimental data, implementating the models, and training. 

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
    *  `DataGen.ipynb` : Script for generating data for the graph coupled examples. May also be used to generate data for the single network examples by setting q=1.
    * `DataGen_Tools.py` : Backend code for generating data.
  * `Models`
    * a Python file containing the class for each model implemented in the paper.
  * `Training`
	  * `Train.ipynb` : Script for training the specified model on the specified dataset.
	  * `Train_Tools.py` : Back-end code for `Train.ipynb`.
  * Plots_Tools.py : A set of functions used to create the different plots seen in the paper.

 * `Data` : Directory for storing the datasets.
 * `Models` : Directory which the data for loss plots, training/model settings are automatically, and the models after each epoch are automatically saved to for each experiment.

## Getting Started
- Download dependencies.
- Add root directory to relevant scripts.
- Set hyperparameters, specify model and dataset, then run `Train.ipynb`
