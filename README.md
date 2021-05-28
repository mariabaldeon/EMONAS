# EMONAS
In this work, we present EMONAS-Net, an Efficient MultiObjective NAS framework for 3D medical image segmentation that optimizes both the segmentation accuracy and size of the network.  EMONAS-Net has two key components, a novel search space that considers the configuration of the micro- and macro-structure of the architecture and a Surrogate-assisted Multiobjective Evolutionary based Algorithm (SaMEA algorithm) that efficiently searches for the best hyperparameter values. The SaMEA algorithm uses the information collected during the initial generations of the evolutionary process to identify the most promising subproblems and select the best performing hyperparameter values during mutation to improve the convergence speed. Furthermore, a Random Forest surrogate model is incorporated to accelerate the fitness evaluation of the candidate architectures.

## Requirements

