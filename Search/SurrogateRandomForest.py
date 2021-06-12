
# Defines the surrogate function for the prediction of the val loss
import pandas as pd
import numpy as np
from numpy import linalg as LA
from sklearn.ensemble import RandomForestRegressor


# Turns all the tested models into two vectors.
def TrainingData(CategoricalGene, models_checked):
    ArrayGen=np.array(CategoricalGene[["alpha", "num_filters","blocks", "node2_inp", "node3_inp", "node4_inp", "ops1", "ops2", "ops3", "ops4", "log_param_count"]])
    #print(ArrayGen.shape)
    fx=np.expand_dims(np.array(models_checked["total_loss_norm"]), axis=1)
    #print(fx.shape)
    return fx, ArrayGen

# builds the random forest with the values of the trained architectures.
def BuildRF(fx, ArrayGen):
    regr = RandomForestRegressor(min_samples_split=5, random_state=0, n_estimators=100)
    regr.fit(ArrayGen, fx.ravel())
    return regr

# Predicts the Mean and Variance for the genotype of the child (childGenotype)
def PredictMeanStd(childCathyper, CategoricalGene, models_checked, LogNumParams):
    fx, ArrayGen=TrainingData(CategoricalGene, models_checked)
    regr=BuildRF(fx, ArrayGen)
    childhyper=childCathyper.copy()
    childhyper.append(LogNumParams)
    childhyper=np.expand_dims(childhyper, axis=0)
    prediction=regr.predict(childhyper)[0]
    pred = np.array([tree.predict(childhyper) for tree in regr]).T
    variance_pred=np.std(pred)
    return prediction, variance_pred
