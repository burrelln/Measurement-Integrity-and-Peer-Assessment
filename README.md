# Measurement-Integrity-and-Peer-Assessment

NOTE: Repo is currently under constructon. With new experimental results being incorporated into the paper, the code here will need to be updated and cleaned up.

This repo contains the implementation of an Agent-Based Model (ABM) of peer assessment that was used to conduct the experiments in the paper "Measurement Integrity in Peer Prediction: A Peer Assessment Case Study" (Burrell and Schoenebeck, 2023). 

See the paper ([arXiv:2108.05521](https://arxiv.org/abs/2108.05521)) for more details about the model and the experiments that were conducted.

To run the simulations, there are a few dependencies: the NetworkX, NumPy, SciPy, and Scikit-learn packages are required for running the experiments and the pandas, Matplotlib, and seaborn packages are required for plotting the results (which is done automatically in most cases). All of these packages are included in the [Anaconda Python Distrbution](https://www.anaconda.com/products/individual).

## Navigating the Repo 

A mini-guide:
- The `data` directory contains all of the `.json` files containing the results from the experiments described in the paper. It includes a script (`make_plots.py`) for recreating the plots in the paper from the `.json` data. 
- The `figures` directory contains all of the `.pdf` files for the plots that appear in the paper.
- The `model_code` directory contains all of the Python modules and scripts needed to run an experiment using the model. It also has several sub-directories:
    - The `mechanisms` directory contains the implementations of the various peer prediction mechanisms that we consider.
    - The `real_data` directory contains the Python scripts that are used to run experiments with real peer grading data (see the paper for details). However, the data itself cannot be made public, so these scripts will raise errors when if they are run.
    - The `results` and `figures` directories store the results of new experiments when they are run. `results` stores `.json` files and `figures` stores `.pdf` plots.
    
If you have questions or see what looks like a bug, let me know!
