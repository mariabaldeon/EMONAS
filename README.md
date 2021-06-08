# EMONAS
In this work, we present EMONAS-Net, an Efficient MultiObjective NAS framework for 3D medical image segmentation that optimizes both the segmentation accuracy and size of the network.  EMONAS-Net has two key components, a novel search space that considers the configuration of the micro- and macro-structure of the architecture and a Surrogate-assisted Multiobjective Evolutionary based Algorithm (SaMEA algorithm) that efficiently searches for the best hyperparameter values. The SaMEA algorithm uses the information collected during the initial generations of the evolutionary process to identify the most promising subproblems and select the best performing hyperparameter values during mutation to improve the convergence speed. Furthermore, a Random Forest surrogate model is incorporated to accelerate the fitness evaluation of the candidate architectures.

![alt text](https://github.com/mariabaldeon/EMONAS/blob/main/images/Fig.%201.%20Structure%20Search%20Space.jpg)

# Requirements
* Python 3.7
* Numpy 1.19.2
* Keras 2.3.1
* Tensorflow 1.14.0
* lhsmdu 1.1
* Pygmo 2.16.1
* Simpleitk 2.0.2

# Datasets
The prostate MR images from the PROMISE12 challenge is available [here](https://promise12.grand-challenge.org/), the hippocampus MR images from the Medical Segmentation Decathlon challenge is available [here](http://medicaldecathlon.com/), and the cardiac MR images from the ACDC dataset is available [here](https://www.creatis.insa-lyon.fr/Challenge/acdc/).   

# Train models
To train and evaluate the models found in our work run:. You must first download the datasets from the links provided and locate them in the folder Dataset/Promise12 for the prostate segmentation, Dataset/ACDC for the cardiac segmentation, and Dataset/MSD for the hippocampus segmentation.    
## Promise 12 dataset

## ACDC dataset

## Hippocampus Medical Declathon dataset

# Architecture search 
To carry out the architecture search run

The output will  be a .csv file with the approximate Pareto Front and a .csv file with all the architectures trained during the optimization process. Select the architecture that best satisfies your requirements. For our experiments, we select the solution in the Pareto Front that minimizes the expected segmentation error. Note the validation performance in this search is not the final performance of the architecture. We only train for a maximum of 120 epochs during the optimization process. You must fully train the architecture from sctrach and the select the weights that miniimizes the validation error. 

Due to the stochastic nature of the search, each run will end with different local minimum architectures. To obtain the best results your must run the search with different seeds and select the architecture that has the best validation performance after fully training it.    
