# Measurement-Integrity-and-Peer-Assessment

This repo contains the implementation of an Agent-Based Model (ABM) of peer assessment that was used to conduct the experiments in the paper "Measurement Integrity in Peer Prediction: A Peer Assessment Case Study" (Burrell and Schoenebeck, 2021). 

See the paper (link forthcoming) for more details about the model and the experiments that were conducted.

To run the simulations, there are a few dependencies: the NetworkX, NumPy, SciPy, and Scikit-learn packages are required for running the experiments and the pandas, Matplotlib, and seaborn packages are required for plotting the results (which is done automatically). All of these packages are included in the [Anaconda Python Distrbution](https://www.anaconda.com/products/individual).

## Navigating the Repo 

A mini-guide:
- The `data` directory contains all of the `.json` files containing the results from the experiments described in the paper. It includes a script for recreating the plots in the paper from the `.json` data. 
- The `figures` directory contains all of the `.pdf` files for the plots that appear in the paper.
- The `model_code` directory contains all of the Python modules and scripts needed to run an experiment using the model.

If you have questions or see what looks like a bug, let me know!
