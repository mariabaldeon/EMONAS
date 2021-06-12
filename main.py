#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import sys
from Datasets.Promise12.DataManager import DataManager
from Search.SaMEA_Algorithm import SaMEA
from Train.model_train import ModelTrain
from Evaluation.evaluation_model import ModelEvaluate

params=dict()
params["DataManager"]=dict()
params["SearchParams"]=dict()
params["TrainParams"]=dict()
params["EvalParams"]=dict()

############ Set Image Preprocessing Parameters ###############################
# Voxel spacing
params["DataManager"]["VolSpa"]=np.asarray([1,1,1.5],dtype=float)
# Fixed size of the image
params["DataManager"]["VolSize"]=np.asarray([128,128,64],dtype=int)
# Total Number of images in the dataset
params["DataManager"]["NumImages"]=50
# Number of validation images
params["DataManager"]["TestImages"]=10
# Path to the dataset
basePath=os.getcwd()
pathset = os.path.join(basePath, "Datasets/Promise12/Images")
params["DataManager"]["dirTrain"]=pathset

############ Set Search Parameters ###############################
# Learning Generations
params["SearchParams"]["LGen"]=10
# Training Epochs for candidate architectures
params["SearchParams"]["epochs"]=120
# Population size
params["SearchParams"]["pop_size"]=10
# neighborhood size
params["SearchParams"]["nei_size"]=3
# Number of generations
params["SearchParams"]["max_gen"]=40
# penalty
params["SearchParams"]["penalty"]=0.001
# patch size to train candidate architectures
params["SearchParams"]["patch_size"]=(128,128,16)
# batch size to train candidate architectures
params["SearchParams"]["batch_size"]=2
# Alpha parameter in expected segmentation error loss function
params["SearchParams"]["alpha"]=0.25
# Beta parameter in expected segmentation error loss function
params["SearchParams"]["beta"]=0.25

############ Set Training Parameters ###############################
# Genotype to decode into EMONAS architecture
# Genotype=[learning_rate,num_filters,num_cells, node2_in, node3_in, node4_in, ops_node1, ops_node2, ops_node3, ops_node4]
params["TrainParams"]["gene"]=[0.0003,32,7,1,0,2, "convP3d_3x3", "conv3d_3x3x3", "conv2d_5x5", "conv3d_1x1x1"]
# image patch size to train architecture
params["TrainParams"]["patch_size"]=(128,128,16)
# Training Epochs
params["TrainParams"]["num_epochs"]=6000
# batch size to train architectures
params["TrainParams"]["batch_size"]=4

############ Set Evaluation Parameters ###############################
params["EvalParams"]["gene"]=[0.003,32,7,1,0,2, "convP3d_3x3", "conv3d_3x3x3", "conv2d_5x5", "conv3d_1x1x1"]
# stride to make the prediction
params["EvalParams"]["stride"]=(128,128,1)
# Path to the weights you want to evaluate
params["EvalParams"]["path"]="weights"

if sys.argv[1]=="-search":
    # Preprocessing images
    DM=DataManager(params["DataManager"])
    DM.preprocess()

    # Architecture Search
    NAS=SaMEA(params["SearchParams"], DM.X_train, DM.X_test, DM.y_train, DM.y_test)
    NAS.search()

elif sys.argv[1]=="-train":
    # Preprocessing images
    DM=DataManager(params["DataManager"])
    DM.preprocess()

    # Train
    model=ModelTrain(params["TrainParams"], DM.X_train, DM.X_test, DM.y_train, DM.y_test)
    model.train()

elif sys.argv[1]=="-evaluate": 
    # Preprocessing images
    DM=DataManager(params["DataManager"])
    DM.preprocess()

    # Evaluate
    EV=ModelEvaluate(params["EvalParams"],params["TrainParams"]["patch_size"],
                params["DataManager"]["VolSpa"], DM.X_test, DM.y_test)
    EV.evaluate()
