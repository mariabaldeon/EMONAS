#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import keras
import logging
import os
from keras import optimizers
from keras.callbacks import CSVLogger,ModelCheckpoint, EarlyStopping, TerminateOnNaN, ReduceLROnPlateau
from keras import backend as K
from Search.EMONASArch import get_EMONAS
from math import sqrt
from numpy import linalg as LA
import math
import timeit
from Search.ImageGenerator_3dcrop import ImageDataGenerator
import lhsmdu
import bisect
import pygmo as pg
from Search.SurrogateRandomForest import PredictMeanStd
from sklearn.metrics import mean_squared_error
from numpy.random import seed
seed(12)
#from tensorflow import set_random_seed
#set_random_seed(12)


# In[11]:


def dice_coef(y_true, y_pred):
    y_true_f=K.flatten(y_true)
    y_pred_f=K.flatten(y_pred)
    intersection=K.sum(y_true_f*y_pred_f)
    return(2.*intersection+0.5)/((K.sum(y_true_f*y_true_f)) + K.sum(y_pred_f*y_pred_f) + 0.5)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

class SaMEA(object):
    def __init__(self,parameters, X_train, X_test, y_train, y_test):

        self.n_hyper=10
        self.pop_size=parameters["pop_size"]
        self.nei_size=parameters["nei_size"]
        self.max_gen=parameters["max_gen"]
        self.penalty=parameters["penalty"]
        self.patch_size=parameters["patch_size"]
        self.batch_size=parameters["batch_size"]
        self.LGen=parameters["LGen"]
        self.w_tloss=parameters["alpha"]
        self.w_eloss=parameters["beta"]
        self.n_epochs=parameters["epochs"]
        self.X_train_r=X_train
        self.X_val_r=X_test
        self.y_train_r=y_train
        self.y_val_r=y_test

        # =========================================================================================================
        # Hyperparameter being Optimized
        # alpha= learning rate
        # num_filters= number of the initial filters. The filters will be doubled in the downsampling and halved in the upsampling size=[8,16,32]
        # blocks= Total number of blocks downsampling+upsampling
        # nodei_inp=input to node i
        # ops= possible operations for each node
        # n_hyper= number of hyperparameters to be changed
        # Genotype=[alpha,num_filters,blocks, node2_in, node3_in, node4_in, ops_node1, ops_node2, ops_node3, ops_node4]
        # =========================================================================================================
        self.alpha= [1e-03,9e-04, 8e-04,7e-04,6e-04, 5e-04, 4e-04, 3e-04, 2e-04, 1e-04,
               9e-05, 8e-05, 7e-05, 6e-05, 5e-05, 4e-05, 3e-05, 2e-05, 1e-05]
        self.num_filters=[32,16,8]
        self.blocks=[9, 7, 5]
        self.node2_inp=[0, 1]
        self.node3_inp=[0, 1, 2]
        self.node4_inp=[0, 1, 2, 3]
        self.ops=['conv3d_1x1x1','conv3d_3x3x3','conv2d_3x3', 'conv2d_5x5','convP3d_3x3',
             'convP3d_5x5', 'identity','conv3d_5x5x5', 'conv2d_7x7','convP3d_7x7']
        # Only these operations will  be avaiable when n_block=9 and n_filter=32 to prevent OOM error
        self.ops_932=['conv3d_1x1x1','conv3d_3x3x3','conv2d_3x3', 'conv2d_5x5','convP3d_3x3',
                 'convP3d_5x5', 'identity']


        #Saves the name of all the hyperparameters and possible hyperparameter values
        self.HypName=["alpha","num_filters", "blocks", "node2_inp", "node3_inp","node4_inp","ops1","ops2", "ops3", "ops4"]
        self.HypVal=[self.alpha,self.num_filters,self.blocks,self.node2_inp,
                     self.node3_inp, self.node4_inp,self.ops, self.ops, self.ops,self.ops]
        self.HypName932=["ops1_932", "ops2_932", "ops3_932", "ops4_932"]
        self.HypVal932=[self.ops_932,self.ops_932,self.ops_932,self.ops_932]

    def log_name(self,generation):
        location="SearchLogs"
        if not os.path.exists(location):
            os.makedirs(location)
        logger=[]
        weights_name=[]
        params=[]
        for i in range(0,self.pop_size):
            name=location+"/"+str(generation)+'_'+str(i)+'_training.log'
            w_name=location+"/"+str(generation)+'_'+str(i)+'_weights.{epoch:02d}-{val_loss:.2f}.hdf5'
            p_name=location+"/"+str(generation)+'_'+str(i)+'_res+alpha.log'
            logger.append(name)
            weights_name.append(w_name)
            params.append(p_name)
        return logger, weights_name, params

    def generate_weight_vectors(self):
        initial_weight=0.5
        increase=(1-0.5)/(self.pop_size-1)
        weight_vectors=[]
        for i in range(self.pop_size):
            w_z1=initial_weight+increase*i
            w_z2=1-w_z1
            weight_vectors.append((w_z1,w_z2))
        return weight_vectors


    def compute_neighbors(self):
        neighbor=np.zeros((len(self.weight_vectors),self.nei_size))

        for j in range(len(self.weight_vectors)):
            weight=self.weight_vectors[j]
            distance = pd.DataFrame(columns=['i','dist'])
            for i in range(len(self.weight_vectors)):
                weight_2=self.weight_vectors[i]
                dist = sqrt( (weight_2[0] - weight[0])**2 + (weight_2[1] - weight[1])**2 )
                distance=distance.append({'i':i,'dist':dist}, ignore_index=True)

            distance=distance.sort_values(by=['dist'])
            dist_m=distance.to_numpy()
            neighbor[j,:]=dist_m[:self.nei_size,0]
        return neighbor

    def SortInitialPop(self, df):
        df=df.sort_values(by=['total_loss', 'param_count'], ascending=[False, True])
        df = df.reset_index(drop=True)
        for index, row in df.iterrows():
            df.loc[index,'Var']='x_0_'+str(index)
            df.loc[index,'subproblem']=index
        return df

    def generate_parents(self, j, OF):

        parents=np.random.choice(self.neighbor[j],2, replace=False)
        parent1=OF.loc[OF['Var']=='x_0_'+str(int(parents[0])),['alpha','num_filters','blocks',  'node2_inp', 'node3_inp',
                                                       'node4_inp','ops1', 'ops2', 'ops3', 'ops4']]
        parent2=OF.loc[OF['Var']=='x_0_'+str(int(parents[1])),['alpha','num_filters','blocks',  'node2_inp', 'node3_inp',
                                                       'node4_inp','ops1', 'ops2', 'ops3', 'ops4']]

        return parent1, parent2

    def recombination(self, parent1, parent2, p):

        child_hyper=[]
        for k in range(self.n_hyper):
            prob=np.random.uniform(0,1)
            if prob<p:
                child_hyper.append(parent1.iloc[0,k])
            if prob>=p :
                child_hyper.append(parent2.iloc[0,k])

        return child_hyper

    def mutation_prob(self, gen):
        fi_0=min(20/self.n_hyper, 1)
        p_n=max(fi_0*(1-(math.log(gen-1+1)/(math.log(self.max_gen)))), 1/self.n_hyper)
        return p_n

    def mutation(self, p, ChildCatHyper, SelectProb):
        for i in range(len(ChildCatHyper)):
            prob=np.random.uniform(0,1)
            if prob<=p:
                mut_index=np.random.choice(len(self.HypVal[i]),1, p=SelectProb[self.HypName[i]])[0]
                while ChildCatHyper[i]==mut_index:
                    mut_index=np.random.choice(len(self.HypVal[i]),1, p=SelectProb[self.HypName[i]])[0]
                ChildCatHyper[i]=mut_index

        child_hyper=[self.alpha[ChildCatHyper[0]],self.num_filters[ChildCatHyper[1]],
                     self.blocks[ChildCatHyper[2]], self.node2_inp[ChildCatHyper[3]],
                     self.node3_inp[ChildCatHyper[4]], self.node4_inp[ChildCatHyper[5]],
                    self.ops[ChildCatHyper[6]], self.ops[ChildCatHyper[7]],
                     self.ops[ChildCatHyper[8]],self.ops[ChildCatHyper[9]]]

        return child_hyper, ChildCatHyper


    # Check if the genotype is correct.
    # Mutate if the operation is too big and can cause OOM problems
    def check_childhyper(self, child_hyper,SelectProb, childCathyper):

        for i in range(len(self.HypName932)):
            # If the operations used in this architectures is bigger than allowed, mutate to a new value allowed
            if child_hyper[6+i] not in self.ops_932:
                mut_index=np.random.choice(len(self.HypVal932[i]),1, p=SelectProb[self.HypName932[i]])[0]
                newhyper=self.HypVal932[i][mut_index]
                child_hyper[6+i]=newhyper
                childCathyper[6+i]=mut_index
        return child_hyper, childCathyper

    def calculate_BI(self, OF, weights):
        OF=np.matrix([[OF[1]],[OF[2]]])
        d1=LA.norm(OF.T*weights)/LA.norm(weights)
        d2=LA.norm(OF-(d1*(weights/LA.norm(weights))))
        ObjFunc=d1+self.penalty*d2

        return ObjFunc

    def initialize_CumProb(self, hyper):
        num=len(hyper)
        probVector=np.zeros(num)
        probVal=1/num
        probCum=0
        for i in range(num-1):
            probCum=probCum+probVal
            probVector[i]=probCum
        # Make sure the last values is 1 as rounding probabilities might end up with decimals lower than 1
        probVector[-1]=1
        return probVector

    def initialize_UniProb(self, size):
        probVector=np.zeros(size)
        probVector=probVector+1/size
        return probVector

    # Initializes the Selection Probability of the Hyperparameters and Subproblem with a uniform prob
    def Initialize_SelectionProb(self):

        SelectProb= dict()
        SelectProb["alpha"]=self.initialize_UniProb(len(self.alpha))
        SelectProb["num_filters"]=self.initialize_UniProb(len(self.num_filters))
        SelectProb["blocks"]=self.initialize_UniProb(len(self.blocks))
        SelectProb["node2_inp"]=self.initialize_UniProb(len(self.node2_inp))
        SelectProb["node3_inp"]=self.initialize_UniProb(len(self.node3_inp))
        SelectProb["node4_inp"]=self.initialize_UniProb(len(self.node4_inp))
        SelectProb["ops1"]=self.initialize_UniProb(len(self.ops))
        SelectProb["ops2"]=self.initialize_UniProb(len(self.ops))
        SelectProb["ops3"]=self.initialize_UniProb(len(self.ops))
        SelectProb["ops4"]=self.initialize_UniProb(len(self.ops))
        SelectProb["ops1_932"]=self.initialize_UniProb(len(self.ops_932))
        SelectProb["ops2_932"]=self.initialize_UniProb(len(self.ops_932))
        SelectProb["ops3_932"]=self.initialize_UniProb(len(self.ops_932))
        SelectProb["ops4_932"]=self.initialize_UniProb(len(self.ops_932))
        SelectProb["subproblem"]=self.initialize_UniProb(self.pop_size)
        return SelectProb

    def Initialize_HyperSelectionProb(self):
        MutHyperProb= dict()

        MutHyperProb["alpha"]=self.initialize_CumProb(self.alpha)
        MutHyperProb["num_filters"]=self.initialize_CumProb(self.num_filters)
        MutHyperProb["blocks"]=self.initialize_CumProb(self.blocks)
        MutHyperProb["node2_inp"]=self.initialize_CumProb(self.node2_inp)
        MutHyperProb["node3_inp"]=self.initialize_CumProb(self.node3_inp)
        MutHyperProb["node4_inp"]=self.initialize_CumProb(self.node4_inp)
        MutHyperProb["ops1"]=self.initialize_CumProb(self.ops)
        MutHyperProb["ops2"]=self.initialize_CumProb(self.ops)
        MutHyperProb["ops3"]=self.initialize_CumProb(self.ops)
        MutHyperProb["ops4"]=self.initialize_CumProb(self.ops)
        MutHyperProb["ops1_932"]=self.initialize_CumProb(self.ops_932)
        MutHyperProb["ops2_932"]=self.initialize_CumProb(self.ops_932)
        MutHyperProb["ops3_932"]=self.initialize_CumProb(self.ops_932)
        MutHyperProb["ops4_932"]=self.initialize_CumProb(self.ops_932)

        return MutHyperProb

    # The gene of an architecture is encoded as gene=[alpha,num_filters,blocks, node2_in, node3_in, node4_in, ops_node1, ops_node2, ops_node3, ops_node4]
    # Fuction that receives the probability of selection of a hyperparameter and chooses the hyperparameter value based
    # on the cumulative probability of selection of a hyperparameter and decodes it into the specific gene.
    def ReturnInitialGene(self, MutProb,MutHyperSelectionProb):

        indAlp=bisect.bisect_left(MutHyperSelectionProb["alpha"], MutProb[0,0])
        indNumFil=bisect.bisect_left(MutHyperSelectionProb["num_filters"], MutProb[0,1])
        indBlock=bisect.bisect_left(MutHyperSelectionProb["blocks"], MutProb[0,2])

        # Decode probabilities to genotype
        alphai=self.alpha[indAlp]
        num_filtersi=self.num_filters[indNumFil]
        blocksi=self.blocks[indBlock]
        nodeinp2=self.node2_inp[bisect.bisect_left(MutHyperSelectionProb["node2_inp"], MutProb[0,3])]
        nodeinp3=self.node3_inp[bisect.bisect_left(MutHyperSelectionProb["node3_inp"], MutProb[0,4])]
        nodeinp4=self.node4_inp[bisect.bisect_left(MutHyperSelectionProb["node4_inp"], MutProb[0,5])]

        # Avoid big architectures because OOM errors
        if num_filtersi==32 and blocksi==9:
            indops1=bisect.bisect_left(MutHyperSelectionProb["ops1_932"], MutProb[0,6])
            indops2=bisect.bisect_left(MutHyperSelectionProb["ops2_932"], MutProb[0,7])
            indops3=bisect.bisect_left(MutHyperSelectionProb["ops3_932"], MutProb[0,8])
            indops4=bisect.bisect_left(MutHyperSelectionProb["ops4_932"], MutProb[0,9])

            ops1=self.ops_932[indops1]
            ops2=self.ops_932[indops2]
            ops3=self.ops_932[indops3]
            ops4=self.ops_932[indops4]
        else:

            indops1=bisect.bisect_left(MutHyperSelectionProb["ops1"], MutProb[0,6])
            indops2=bisect.bisect_left(MutHyperSelectionProb["ops2"], MutProb[0,7])
            indops3=bisect.bisect_left(MutHyperSelectionProb["ops3"], MutProb[0,8])
            indops4=bisect.bisect_left(MutHyperSelectionProb["ops4"], MutProb[0,9])
            ops1=self.ops[indops1]
            ops2=self.ops[indops2]
            ops3=self.ops[indops3]
            ops4=self.ops[indops4]

        gene=[alphai,num_filtersi,blocksi ,nodeinp2,nodeinp3,nodeinp4, ops1,ops2,ops3, ops4]

        cat_gene=[indAlp, indNumFil, indBlock,nodeinp2,nodeinp3,nodeinp4, indops1, indops2, indops3, indops4]
        return gene, cat_gene

    # Selects subproblem according to selection probabilities
    def SelectSuproblem(self,SelectProb, i, j):
        if i<=self.LGen:
            return j
        else:
            return np.random.choice(self.pop_size,1, p=SelectProb["subproblem"])[0]

    def SolutionsToList(self, models_checked):
        sol_list=[]
        for i in range(len(models_checked)):
            lossi=models_checked['total_loss_norm'].iloc[i]
            parami=models_checked['log_param_count_norm'].iloc[i]
            sol_list.append([lossi,parami])
        return sol_list

    def BestSolutionsNSGA(self, models_checked):
        sol_list= self.SolutionsToList(models_checked)
        best_sol=pg.select_best_N_mo(points = sol_list, N=self.pop_size)
        PolNSGA=pd.DataFrame()
        for i in range(len(best_sol)):
            PolNSGA=PolNSGA.append(models_checked.iloc[best_sol[i]])
        return PolNSGA

    # Updates the SucessSolMatrix that counts the the number of sucessful solutions (part of NSGA best solutions) generated by each suproblem (column)
    # and in each generation (rows)
    def Update_SucessSolMatrix(self,Gen,PolNSGA,SucessSolMatrix):
        Gen_NSGA=PolNSGA[PolNSGA['Var'].str.contains('x_'+str(Gen))]['subproblem']
        for subprob in Gen_NSGA:
            SucessSolMatrix[Gen,int(subprob)]=SucessSolMatrix[Gen,int(subprob)]+1
        return SucessSolMatrix

    # Updates the Subproblem Selection probability based on the NSGA optimal population
    def UpdateSubproblemSelProb(self, SucessSolMatrix, Gen, PolNSGA, SelectProb):
        SucessSolMatrix=self.Update_SucessSolMatrix(Gen,PolNSGA,SucessSolMatrix)
        if Gen>self.LGen:
            NumSucessSubP=np.sum(SucessSolMatrix[:Gen+1,:], axis=0)
            TotalNumSucess=np.sum(NumSucessSubP)
            eps=0.2*(1/self.pop_size)
            ProportionSuccess=np.divide(NumSucessSubP,TotalNumSucess+1e-6)+eps
            TotalProportionSuccess=np.sum(ProportionSuccess)
            SelectProb["subproblem"]=np.divide(ProportionSuccess,TotalProportionSuccess)
        return SucessSolMatrix, SelectProb

    # Computes the selection probability for each hyperparameter value by considering the average performance in other candidates
    def ComputeSelectProbHyper(self, PerfMean,NumHyperValues):
        TotalPerfMean=np.sum(PerfMean)
        eps=0.20*(1/NumHyperValues)
        PerfProportion=np.divide(PerfMean,TotalPerfMean)+eps
        TotalProportionPerf=np.sum(PerfProportion)
        return np.divide(PerfProportion,TotalProportionPerf)

    # Updates the Hyperparameters Selection probability based on the historic accuracy performance
    def UpdateHyperSelProb(self, Gen, SelectProb,CategoricalGene):
        if Gen>self.LGen:
            for i in range(len(self.HypName)):
                NumHyperValues=len(self.HypVal[i])
                PerfMean=np.zeros((NumHyperValues))
                for j in range(NumHyperValues):
                    MeanAcc=self.max_OF[1]-np.mean(CategoricalGene[CategoricalGene[self.HypName[i]]==j]["total_loss"])
                    if not math.isnan(MeanAcc):
                        PerfMean[j]=MeanAcc
                SelectProb[self.HypName[i]]=self.ComputeSelectProbHyper(PerfMean,NumHyperValues)

            for k in range(len(self.HypName932)):
                NumHyperValues=len(self.HypVal932[k])
                PerfMean=np.zeros((NumHyperValues))
                for m in range(NumHyperValues):
                    MeanAcc=self.max_OF[1]-np.mean(CategoricalGene[CategoricalGene[self.HypName[6+k]]==m]["total_loss"])
                    if not math.isnan(MeanAcc):
                        PerfMean[m]=MeanAcc
                SelectProb[self.HypName932[k]]=self.ComputeSelectProbHyper(PerfMean,NumHyperValues)
        return SelectProb

    def FindParetoSol(self, models_checked):
        models_checked['log_param_count_norm']=(models_checked['log_param_count']-self.Z_ref[2])/(self.max_OF[2]-self.Z_ref[2])
        sol_list= self.SolutionsToList(models_checked)
        pareto_sol=pg.non_dominated_front_2d(points = sol_list)
        Par=pd.DataFrame()
        for i in range(len(pareto_sol)):
            Par=Par.append(models_checked.iloc[pareto_sol[i]])
        return Par, pareto_sol

    #Computes the pareto front from all the population checked and returns dataframe
    def ParetoFront(self, models_checked):
        models_checked['total_loss_norm']=(models_checked['total_loss']-self.Z_ref[1])/(self.max_OF[1]-self.Z_ref[1])
        models_checked['log_param_count_norm']=(models_checked['log_param_count']-self.Z_ref[2])/(self.max_OF[2]-self.Z_ref[2])
        sol_list= self.SolutionsToList(models_checked)
        pareto_sol=pg.non_dominated_front_2d(points = sol_list)
        Par=pd.DataFrame()
        for i in range(len(pareto_sol)):
            Par=Par.append(models_checked.iloc[pareto_sol[i]])
        return Par, pareto_sol

    def ParetoObjFunc_List(self,SolList,ParetoSolList):
        ParetoObjFuncList=[]
        for i in range(len(ParetoSolList)):
            ObjFunci=SolList[ParetoSolList[i]]
            ParetoObjFuncList.append(ObjFunci)
        return ParetoObjFuncList

    def next_batch(self, X, y, size):
        while True:
            perm=np.random.permutation(X.shape[0])
            for i in np.arange(0, X.shape[0], size):
                Xi=X[perm[i:i+size]]
                yi=y[perm[i:i+size]]
                yield Xi, yi

    def calculate_reference(self):
        input_list=[1,2,3]
        #The smallest architecture is just made by identity operations
        op_list=['identity','identity','identity','identity']
        model=get_EMONAS(input_list,op_list, h=self.patch_size[0], w=self.patch_size[1],
                           slices=self.patch_size[2], channels=self.channels, classes=self.classes, nfilter=int(np.min(self.num_filters)),
                                  blocks=int(np.min(self.blocks)))
        min_parameters=model.count_params()

        Z1_min=0
        Z2_min=0
        Z3_min=np.log(min_parameters)

        return [Z1_min,Z2_min,Z3_min]

    def calculate_maxpoint(self):
        # input that maximizes the size
        input_list=[1,2,3]
        #The biggest architecture is made by 3d 5x5x5 convolutions
        # Recall 9 blocks, 32 filters and conv3d5x5x5 is not permited. Choose the biggest afterwards
        op_list=['conv3d_5x5x5','conv3d_5x5x5','conv3d_5x5x5','conv3d_5x5x5']
        model=get_EMONAS(input_list,op_list, h=self.patch_size[0], w=self.patch_size[1],
                           slices=self.patch_size[2], channels=self.channels, classes=self.classes, nfilter=16,
                                  blocks=int(np.max(self.blocks)))
        max_parameters=model.count_params()
        max_total_loss=self.classes+self.classes*self.w_tloss+self.w_eloss*1

        return [self.classes,max_total_loss,np.log(max_parameters)]

    # Expected Segmentatio Error loss function
    def total_loss_fc(self,train_loss, val_loss, min_loss):
        return self.w_tloss*train_loss+val_loss+self.w_eloss*((self.n_epochs-min_loss)/self.n_epochs)

    #Genotype=[alpha,num_filters,blocks, node2_in, node3_in, node4_in, ops_node1, ops_node2, ops_node3, ops_node4]
    # Trains candidates architectures
    def model_train_bp(self, generation, gene, logger, weights_name, params, indv):
        input_list=[gene[3],gene[4],gene[5]]
        op_list=[gene[6],gene[7],gene[8],gene[9]]
        model=get_EMONAS(input_list, op_list, h=self.patch_size[0], w=self.patch_size[1],
                       slices=self.patch_size[2], channels=self.channels, classes=self.classes, nfilter=gene[1],blocks=gene[2])
        model.summary()
        adam=optimizers.Adam(lr=gene[0], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss=dice_coef_loss, optimizer=adam)
        csv_logger = CSVLogger(logger)
        model_check=ModelCheckpoint(filepath= weights_name , monitor='val_loss', verbose=0, save_best_only=True)
        early_stopper=EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20, mode='auto')
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,patience=30, mode='min', min_lr=1e-07)
        logging.basicConfig(filename=params, level=logging.INFO)
        history=model.fit_generator(self.train_generator, steps_per_epoch=(self.X_train_r.shape[0]/self.batch_size),
                                    validation_data=self.next_batch(self.X_val_r, self.y_val_r, 2),
                                    validation_steps=(self.y_val_r.shape[0]/2), epochs=self.n_epochs,
                                    callbacks=[csv_logger, early_stopper, reduce_lr])
        if math.isnan(history.history['loss'][-1]):
            validation_loss=1000
            train_loss=1000
        elif math.isnan(history.history['val_loss'][-1]):
            validation_loss=1000
            train_loss=1000
        else :
            validation_loss=np.mean(history.history['val_loss'][-5:])
            train_loss=np.mean(history.history['loss'][-5:])
        train_parameters=model.count_params()
        min_index=np.argmin(history.history['val_loss'])
        total_epochs=len(history.history['val_loss'])
        del model
        K.clear_session()
        return train_loss, validation_loss, train_parameters, total_epochs, min_index

    def CountParams(self,child_hyper):
        input_list=[child_hyper[3],child_hyper[4],child_hyper[5]]
        op_list=[child_hyper[6],child_hyper[7],child_hyper[8],child_hyper[9]]
        model=get_EMONAS(input_list, op_list, h=self.patch_size[0], w=self.patch_size[1],
                       slices=self.patch_size[2], channels=self.channels, classes=self.classes, nfilter=child_hyper[1],blocks=child_hyper[2])
        return float(model.count_params())

    def val_stride(self,img_dim, patch_dim):
        total_patch=math.ceil(img_dim/patch_dim)
        if total_patch==1:
            return img_dim, total_patch
        pix_dif=(patch_dim*total_patch)-img_dim
        stride_dif=math.ceil(pix_dif/(total_patch-1))
        stride=patch_dim-stride_dif
        return stride, total_patch

    def val_convert_patch(self,X_val, patch_dim):
        num, row, col, sl, ch= X_val.shape
        pt_row, pt_col, pt_sl= patch_dim
        row_str, num_row=self.val_stride(row, pt_row)
        col_str, num_col=self.val_stride(col, pt_col)
        sl_str, num_sl=self.val_stride(sl, pt_sl)
        img_patch=num_row*num_col*num_sl
        total_patch=num*img_patch
        X_val_patch=np.zeros((total_patch, pt_row, pt_col, pt_sl, ch))
        ix_patch=0

        for i in range(num):
            for j in range(num_row):
                for k in range(num_col):
                    for m in range(num_sl):
                        row_in=j*row_str
                        col_in=k*col_str
                        sl_in=m*sl_str
                        row_fin=row_in+pt_row
                        col_fin=col_in+pt_col
                        sl_fin=sl_in+pt_sl
                        X_val_patch[ix_patch,:,:,:,:]=X_val[i,row_in:row_fin,col_in:col_fin,sl_in:sl_fin,:]
                        ix_patch=ix_patch+1
        return X_val_patch

    # Function that trains the selected candidates and compares them with the neighborhood
    def TrainSelectedCandidates(self,TrainCandidateList,PossibleTrainCandidates,Gen,models_checked,
                CategoricalGene, genotype, OF, logger, weights_name, params):

        for indv in TrainCandidateList:
            CandidateParameters=PossibleTrainCandidates[["child_hyper","childCathyper", "subproblem",
                                    "predmean", "predstd"]].loc[PossibleTrainCandidates["indv"]==indv].values.tolist()
            child_hyper,childCathyper,SPj, predmean, predstd =CandidateParameters[0]

            train_loss, validation_loss, train_params, total_epochs, min_index= self.model_train_bp(Gen, child_hyper, logger[indv],
                                            weights_name[indv], params[indv],indv)

            total_loss=self.total_loss_fc(train_loss, validation_loss, min_index)

            models_checked, CategoricalGene= self.save_dataframe_info(models_checked, CategoricalGene,Gen,indv,
                                child_hyper, train_loss,validation_loss,train_params, total_epochs, min_index,SPj,
                                                    childCathyper, total_loss)

            genotype.append(childCathyper)

            child_OF=[(train_loss-self.Z_ref[0])/(self.max_OF[0]-self.Z_ref[0]),(total_loss-self.Z_ref[1])/(self.max_OF[1]-self.Z_ref[1]),
                                      (np.log(train_params)-self.Z_ref[2])/(self.max_OF[2]-self.Z_ref[2])]

            for m in range(self.nei_size):
                ObjFunc_nei, ObjFunc_chi=self.ReturnLoss(child_OF, SPj, m, OF)
                if ObjFunc_chi<=ObjFunc_nei:
                    OF= self.ReplaceOFSolution(OF, self.neighbor[SPj][m], childCathyper, train_loss, validation_loss, train_params,
                                                      total_epochs,min_index,SPj,child_OF, Gen, indv, total_loss)
        return models_checked, CategoricalGene, genotype, OF

    # Selects which candidates architectures to train (Pareto solutions, minimium loss, max prediction variability)
    def ReturnCandidatesToTrain(self, pareto_solutions,PossibleTrainCandidates):
        CurrentParetoSol=pareto_solutions.copy()
        PosTrainCandidate=PossibleTrainCandidates.copy()

        CurrentParetoSol=pd.concat([CurrentParetoSol,PosTrainCandidate.rename(columns={"indv": "Var", "predmean":"total_loss_norm" })])
        NewParetoSol, ParetoList=self.FindParetoSol(CurrentParetoSol)
        TrainCandidateList=list(NewParetoSol.dropna(how='any', subset=["childCathyper"])["Var"])

        for i in range(len(TrainCandidateList)):
            PosTrainCandidate=PosTrainCandidate[PosTrainCandidate["indv"]!=TrainCandidateList[i]]

        if not PosTrainCandidate.empty:
            MinLossCandidate=PosTrainCandidate["indv"].loc[PosTrainCandidate["predmean"].idxmin()]
            PosTrainCandidate=PosTrainCandidate[PosTrainCandidate["indv"]!=MinLossCandidate]
            TrainCandidateList.append(MinLossCandidate)

        if not PosTrainCandidate.empty:
            MaxStdCandidate=PosTrainCandidate["indv"].loc[PosTrainCandidate["predstd"].idxmax()]
            TrainCandidateList.append(MaxStdCandidate)
        TrainCandidateList=sorted(TrainCandidateList)
        return TrainCandidateList

    def ReplaceOFSolution(self,OF,neighbor_child, childCathyper, train_loss, validation_loss, train_params, total_epochs,
                          min_index,SPj,child_OF, i, j, total_loss):
        neighbor_child=int(neighbor_child)
        OF=OF[OF["Var"]!='x_0_'+str(neighbor_child)]
        OF=OF.append({'Var':'x_0_'+str(neighbor_child),'alpha': childCathyper[0],
                                    'num_filters': childCathyper[1], 'blocks': childCathyper[2], 'node2_inp': childCathyper[3],
                                    'node3_inp': childCathyper[4], 'node4_inp': childCathyper[5], 'ops1': childCathyper[6],
                                    'ops2': childCathyper[7], 'ops3': childCathyper[8],'ops4': childCathyper[9],
                                    'train_loss': train_loss,'val_loss':validation_loss,'param_count':train_params ,
                                    'log_param_count':np.log(train_params),'total_epochs':total_epochs,
                                    'min_epoch_loss':min_index, 'total_loss':total_loss, 'subproblem':SPj, 'train_loss_norm':child_OF[0],
                        'val_loss_norm':validation_loss, 'total_loss_norm':child_OF[1],'log_param_count_norm': child_OF[2],
                            'Real_Var': 'x_'+str(i)+'_'+str(j)}, ignore_index=True)
        OF=OF.sort_values(by=['Var'])
        return OF

    # Returns the loss of the neighbor and child using the Boundary Intersection Approach
    def ReturnLoss(self, child_OF, SPj, m, OF):

        neighbor_child=int(self.neighbor[SPj][m])
        nei_weights=np.asarray(self.weight_vectors[neighbor_child]).reshape((2,1))
        OF_neighbor=OF.loc[OF['Var']=='x_0_'+str(neighbor_child),['train_loss_norm',
                                                                  'total_loss_norm','log_param_count_norm']]
        OF_nei=[OF_neighbor.iloc[0,0],OF_neighbor.iloc[0,1],OF_neighbor.iloc[0,2]]
        ObjFunc_nei=self.calculate_BI(OF_nei,nei_weights)
        ObjFunc_chi=self.calculate_BI(child_OF, nei_weights)
        return ObjFunc_nei, ObjFunc_chi

    # Evaluates the Canidadte architecture.
    # If before LGen or is predicted to be an OF solution it is truly trained and returns true values
    # If not trained, returns the hyperparameter values to be evaluated if it will be trained later based on the whole generation
    def EvaluateCandidateArchitecture(self, Gen, child_hyper,j, childCathyper, CategoricalGene, SPj,
        PossibleTrainCandidates,models_checked, logger, weights_name, params, OF):

        # If the current generation is a learning generations train the candidate architecture
        if Gen<=self.LGen:
            Train=True
            train_loss, validation_loss, train_params, total_epochs, min_index= self.model_train_bp(Gen, child_hyper,
                                                                        logger[j],weights_name[j], params[j],j)
            return (Train, [train_loss, validation_loss, train_params, total_epochs, min_index, "NoPred", "NoStd"])

        else:
            # Predict the mean (total loss norm) and standard deviation using the surrogate function
            NumParams=self.CountParams(child_hyper)
            predmean, predstd=PredictMeanStd(childCathyper, CategoricalGene, models_checked, np.log(NumParams))
            PredChildOF=[0, predmean, (np.log(NumParams)-self.Z_ref[2])/(self.max_OF[2]-self.Z_ref[2])]
            Train=False

            for m in range(self.nei_size):
                #Calculate Boundary Intersection Approach loss of the neighbor and predicted child
                ObjFunc_nei, ObjFunc_chi=self.ReturnLoss(PredChildOF, SPj, m, OF)
                if ObjFunc_chi<=ObjFunc_nei:
                    Train=True
                    #Since ObjFunc_chi<=ObjFunc_nei Train
                    break

            # If the predicted child architecture is a solution than train it
            if Train:
                train_loss, validation_loss, train_params, total_epochs, min_index= self.model_train_bp(Gen, child_hyper, logger[j],
                                        weights_name[j], params[j],j)
                return (Train, [train_loss, validation_loss, train_params, total_epochs, min_index,predmean, predstd])


            else:
                # Add the child architecture to dataframe of possible candidates to be trianed later
                PossibleTrainCandidates= PossibleTrainCandidates.append({'child_hyper':child_hyper, 'childCathyper':childCathyper,
                    'predmean':predmean,'predstd':predstd ,'log_param_count':np.log(NumParams), 'indv':j, 'subproblem': SPj}, ignore_index=True)
                return (Train, PossibleTrainCandidates)


    def save_dataframe_info(self, models_checked, CategoricalGene, Gen, indv,child_hyper, train_loss,validation_loss,
                train_params, total_epochs, min_index,SPj, CatGene, total_loss):

        models_checked = models_checked.append({'Var':'x_'+str(Gen)+'_'+str(indv),'alpha': child_hyper[0],
                                    'num_filters': child_hyper[1], 'blocks': child_hyper[2], 'node2_inp': child_hyper[3],
                                    'node3_inp': child_hyper[4], 'node4_inp': child_hyper[5], 'ops1': child_hyper[6],
                                    'ops2': child_hyper[7], 'ops3': child_hyper[8],'ops4': child_hyper[9],
                                    'train_loss': train_loss,'val_loss':validation_loss,'param_count':train_params ,
                                    'log_param_count':np.log(train_params),'total_epochs':total_epochs,
                                    'min_epoch_loss':min_index, 'total_loss':total_loss,  'subproblem':SPj }, ignore_index=True)

        CategoricalGene = CategoricalGene.append({'Var':'x_'+str(Gen)+'_'+str(indv),'alpha': CatGene[0],'num_filters': CatGene[1],
                                'blocks': CatGene[2], 'node2_inp': CatGene[3], 'node3_inp': CatGene[4], 'node4_inp': CatGene[5],
                            'ops1': CatGene[6], 'ops2': CatGene[7], 'ops3': CatGene[8], 'ops4': CatGene[9],
                                    'train_loss': train_loss,'val_loss':validation_loss,'param_count':train_params ,
                                    'log_param_count':np.log(train_params),'total_epochs':total_epochs,
                                    'min_epoch_loss':min_index,'total_loss':total_loss, 'subproblem':SPj }, ignore_index=True)

        return models_checked, CategoricalGene

    def save_pareto_front(self, pareto_solutions):
        pareto_save=pareto_solutions[["alpha", "node2_inp", "node3_inp","node4_inp", "ops1",
        "ops2", "ops3", "ops4", "blocks", "num_filters", "total_loss", "val_loss", "train_loss", "param_count"]].copy().reset_index()
        pareto_save.rename(columns={"alpha":"learning_rate", "blocks":"num_cells"}, inplace=True)
        pareto_save.to_csv('pareto_solutions.csv')
        self.pareto_front=pareto_save

    def search(self):

        _, height, width, slices, self.channels=self.X_train_r.shape
        self.classes=self.y_val_r.shape[-1]

        # Crop the validation data according to patch
        self.X_val_r=self.val_convert_patch(self.X_val_r, self.patch_size)
        self.y_val_r=self.val_convert_patch(self.y_val_r, self.patch_size)

        datagenX = ImageDataGenerator(rotation_range=90, width_shift_range=0.4, height_shift_range=0.4, zoom_range=0.5,
                                      horizontal_flip=True, data_format='channels_last', random_crop=self.patch_size)
        datagenY = ImageDataGenerator(rotation_range=90, width_shift_range=0.4, height_shift_range=0.4, zoom_range=0.5,
                                      horizontal_flip=True, data_format='channels_last', random_crop=self.patch_size)

        # Provide the same seed and keyword arguments to the fit and flow methods
        seed = 1

        image_generator = datagenX.flow(self.X_train_r, batch_size=self.batch_size, seed=seed)
        mask_generator = datagenY.flow(self.y_train_r, batch_size=self.batch_size, seed=seed)
        self.train_generator = zip(image_generator, mask_generator)

        start_time_algo = timeit.default_timer()

        # Initialize the algorithm
        # Generate initial population using Latin Hypercube Sampling with multi-dimensional uniformity
        IniGenotypeProb = lhsmdu.sample(self.pop_size, self.n_hyper)
        MutHyperSelectionProb=self.Initialize_HyperSelectionProb()

        # Initiliaze a dictionary with a uniforms selection probability for subproblem and hyperparameters
        SelectProb=self.Initialize_SelectionProb()

        #List that saves all the generated genotypes
        genotype=[]

        models_checked = pd.DataFrame(columns=['Var','alpha','num_filters','blocks',  'node2_inp', 'node3_inp', 'node4_inp',
                                                'ops1', 'ops2', 'ops3', 'ops4','train_loss',
                                               'val_loss','param_count','log_param_count','total_epochs', 'min_epoch_loss',
                                              'total_loss', 'subproblem'])
        CategoricalGene = pd.DataFrame(columns=['Var','alpha','num_filters','blocks',  'node2_inp', 'node3_inp', 'node4_inp',
                                                'ops1', 'ops2', 'ops3', 'ops4','train_loss',
                                               'val_loss','param_count','log_param_count','total_epochs', 'min_epoch_loss',
                                              'total_loss', 'subproblem'])

        #Logger info for generation 0
        logger0, weights_name0, params0=self.log_name(0)

        self.Z_ref=self.calculate_reference()
        self.max_OF=self.calculate_maxpoint()

        #Creates the initial population
        for i in range(0,self.pop_size):
            IndviProb=IniGenotypeProb[i]
            gene, CatGene=self.ReturnInitialGene(IndviProb,MutHyperSelectionProb)
            train_loss, validation_loss, train_params, total_epochs, min_index =self.model_train_bp(0, gene,
                                                                        logger0[i], weights_name0[i], params0[i], i)
            # Compute total loss
            total_loss=self.total_loss_fc(train_loss, validation_loss, min_index)
            genotype.append(CatGene)

            models_checked, CategoricalGene= self.save_dataframe_info(models_checked, CategoricalGene,0,i,gene,
                train_loss,validation_loss,train_params, total_epochs, min_index,i, CatGene,total_loss)

        # Sort the initial population such that the architecture with smallest loss is with weights (1,0)
        models_checked=self.SortInitialPop(models_checked)
        CategoricalGene=self.SortInitialPop(CategoricalGene)

        # Calculate the first pareto solutions
        pareto_solutions, ParetoSolList=self.ParetoFront(models_checked)
        pareto_solutions.to_csv('pareto_solutions.csv')

        start=1

        self.weight_vectors=self.generate_weight_vectors()
        self.neighbor=self.compute_neighbors()

        #Evolution!
        # Dataframe that saves the best solutions using MOEA/D PBI approach
        OF=CategoricalGene.copy()
        OF['Real_Var']=OF['Var']

        SucessSolMatrix=np.zeros((self.max_gen,self.pop_size))
        for i in range(start,self.max_gen):
            print("\n")
            print("  -----------------------GENERATION ", i, "  -----------------------")
            print("\n")

            start_time = timeit.default_timer()

            #Adaptive Normalization
            OF['train_loss_norm']=OF['train_loss']
            OF['val_loss_norm']=OF['val_loss']
            OF['total_loss_norm']=(OF['total_loss']-self.Z_ref[1])/(self.max_OF[1]-self.Z_ref[1])
            OF['log_param_count_norm']=(OF['log_param_count']-self.Z_ref[2])/(self.max_OF[2]-self.Z_ref[2])

            #Logger info
            logger, weights_name, params=self.log_name(generation=i)
            # Mutation probability
            prob=self.mutation_prob(i+1)

            # Dataframe with all the candidate architectures that have not been trained and will later be assesed if it should be trained
            PossibleTrainCandidates=pd.DataFrame(columns=['child_hyper','childCathyper','predmean','predstd',
                                                          'log_param_count','indv', 'subproblem'])
            for j in range(self.pop_size):

                print("Evaluating candidate architecture ", j)

                #Select the subproblem SPj
                SPj=self.SelectSuproblem(SelectProb, i, j)
                parent1, parent2=self.generate_parents(SPj, OF)
                childCathyper=self.recombination(parent1, parent2, 1/2)
                child_hyper, childCathyper=self.mutation(prob, childCathyper, SelectProb)

                #Assures the same models are not trained
                while childCathyper in genotype:
                    child_hyper, childCathyper=self.mutation(prob, childCathyper, SelectProb)

                # If the child has 9 blocks and 32 filters, make sure the operations are permitted to avoid OOM error
                if child_hyper[1]==32 and child_hyper[2]==9:
                    child_hyper, childCathyper=self.check_childhyper(child_hyper,SelectProb, childCathyper)

               # Evaluate if candidate should be trained
                Train, Result=self.EvaluateCandidateArchitecture(i, child_hyper,j, childCathyper,CategoricalGene,
                    SPj, PossibleTrainCandidates, models_checked, logger, weights_name, params, OF)
                if Train:
                    train_loss, validation_loss, train_params, total_epochs, min_index, predmean, predstd=Result
                    total_loss=self.total_loss_fc(train_loss, validation_loss, min_index)
                    models_checked, CategoricalGene= self.save_dataframe_info(models_checked, CategoricalGene,i,j,
                                    child_hyper, train_loss, validation_loss,train_params, total_epochs, min_index,
                                    SPj, childCathyper, total_loss)
                    genotype.append(childCathyper)
                    child_OF=[(train_loss-self.Z_ref[0])/(self.max_OF[0]-self.Z_ref[0]),(total_loss-self.Z_ref[1])/(self.max_OF[1]-self.Z_ref[1]),
                      (np.log(train_params)-self.Z_ref[2])/(self.max_OF[2]-self.Z_ref[2])]

                    #Calculate BI OF of each neighbor and compare with the child
                    for m in range(self.nei_size):
                        ObjFunc_nei, ObjFunc_chi=self.ReturnLoss(child_OF, SPj, m, OF)
                        if ObjFunc_chi<=ObjFunc_nei:
                            OF= self.ReplaceOFSolution(OF,self.neighbor[SPj][m], childCathyper, train_loss, validation_loss, train_params,
                                                  total_epochs,min_index,SPj,child_OF, i, j, total_loss)
                else:
                    #Retrieve the Dataframe with the candidates that have not been trained
                    PossibleTrainCandidates=Result

                pareto_solutions, ParetoSolList=self.ParetoFront(models_checked)
                self.save_pareto_front(pareto_solutions)
                #models_checked.to_csv('models_checked.csv')

            # After all the individuals in the generation have been generated select candidates with min loss,
            # high variance and pareto solutions to train from the Dataframe PossibleTrainCandidates
            if not PossibleTrainCandidates.empty:
                TrainCandidateList=self.ReturnCandidatesToTrain(pareto_solutions,PossibleTrainCandidates)
                models_checked, CategoricalGene, genotype, OF=self.TrainSelectedCandidates(TrainCandidateList,PossibleTrainCandidates,
                    i,models_checked, CategoricalGene, genotype, OF, logger, weights_name, params)

            # Update populations and probabilities
            pareto_solutions, ParetoSolList=self.ParetoFront(models_checked)
            PolNSGA=self.BestSolutionsNSGA(models_checked)
            SucessSolMatrix, SelectProb=self.UpdateSubproblemSelProb(SucessSolMatrix, i, PolNSGA, SelectProb)
            SelectProb=self.UpdateHyperSelProb(i, SelectProb,CategoricalGene)
            elapsed = timeit.default_timer() - start_time
            logging.info('generation time: %s', str(elapsed))

        #Save info
        final_time=timeit.default_timer()-start_time_algo
        logging.info('Time Elapsed: %s', str(final_time))
        #models_checked.to_csv('models_checked.csv')
        pareto_solutions, ParetoSolList=self.ParetoFront(models_checked)
        self.save_pareto_front(pareto_solutions)
