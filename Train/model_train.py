#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import keras
import logging
from keras import optimizers
from keras.callbacks import CSVLogger,ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN 
from keras import backend as K
from Search.EMONASArch import get_EMONAS
import math
import timeit
from Search.ImageGenerator_3dcrop import ImageDataGenerator


# In[ ]:


def dice_coef(y_true, y_pred):
    y_true_f=K.flatten(y_true)
    y_pred_f=K.flatten(y_pred)
    intersection=K.sum(y_true_f*y_pred_f)
    return(2.*intersection+0.5)/((K.sum(y_true_f*y_true_f)) + K.sum(y_pred_f*y_pred_f) + 0.5)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

def recall(y_true, y_pred):
    y_true_f= K.flatten(y_true)
    y_pred_f= K.flatten(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

class ModelTrain(object): 
    def __init__(self,parameters, X_train, X_val, y_train, y_val):
        self.gene=parameters["gene"]
        self.patch_size=parameters["patch_size"]
        self.num_epochs=parameters["num_epochs"]
        self.batch_size=parameters["batch_size"]
        self.X_train_r= X_train
        self.X_val_r= X_val
        self.y_train_r= y_train
        self.y_val_r=y_val
    
    def val_stride(self, img_dim, patch_dim):
        total_patch=math.ceil(img_dim/patch_dim)
        if total_patch==1: 
            return img_dim, total_patch

        pix_dif=(patch_dim*total_patch)-img_dim
        stride_dif=math.ceil(pix_dif/(total_patch-1))
        stride=patch_dim-stride_dif
        return stride, total_patch
    
    def val_convert_patch(self, X_val):
        num, row, col, sl, ch= X_val.shape
        pt_row, pt_col, pt_sl= self.patch_size
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
                        X_val_patch[ix_patch,:,:,:,0]=X_val[i,row_in:row_fin,col_in:col_fin,sl_in:sl_fin,0]
                        ix_patch=ix_patch+1
        return X_val_patch
    
    def train(self): 
        _, self.height, self.width, self.slices, self.channels=self.X_train_r.shape
        self.classes=self.y_val_r.shape[-1]
        
        # Crop the validation data according to patch
        self.X_val_r=self.val_convert_patch(self.X_val_r)
        self.y_val_r=self.val_convert_patch(self.y_val_r)
        
        datagenX = ImageDataGenerator(rotation_range=90, width_shift_range=0.4, height_shift_range=0.4, zoom_range=0.5, 
                              horizontal_flip=True, data_format='channels_last', random_crop=self.patch_size)
        datagenY = ImageDataGenerator(rotation_range=90, width_shift_range=0.4, height_shift_range=0.4, zoom_range=0.5, 
                                      horizontal_flip=True, data_format='channels_last', random_crop=self.patch_size)


        seed = 1

        image_generator = datagenX.flow(self.X_train_r, batch_size=self.batch_size, seed=seed)
        mask_generator = datagenY.flow(self.y_train_r, batch_size=self.batch_size, seed=seed)
        train_generator = zip(image_generator, mask_generator)
        
        # Training logs 
        location="TrainLogs"
        if not os.path.exists(location):
            os.makedirs(location)
        logger=location+'/training.log'
        weights_name=location+'/weights.{epoch:02d}-{val_dice_coef:.2f}.hdf5'
        params=location+'/res+alpha.log'
        
        start_time = timeit.default_timer()

        input_list=[self.gene[3],self.gene[4],self.gene[5]]
        op_list=[self.gene[6],self.gene[7],self.gene[8],self.gene[9]]
        model=get_EMONAS(input_list, op_list, h=self.patch_size[0], w=self.patch_size[1], 
                       slices=self.patch_size[2], channels=self.channels, 
                         classes=self.classes, nfilter=self.gene[1],blocks=self.gene[2])
        model.summary()
        adam=optimizers.Adam(lr=self.gene[0], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss=dice_coef_loss, optimizer=adam, metrics=['accuracy',
                                                                    dice_coef, 
                                                                    recall])

        csv_logger = CSVLogger(logger)
        model_check=ModelCheckpoint(filepath= weights_name, monitor='val_loss', verbose=0, save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,patience=30, mode='min', min_lr=1e-08)
        logging.basicConfig(filename=params, level=logging.INFO)

        history=model.fit_generator(train_generator, steps_per_epoch=(self.X_train_r.shape[0]/self.batch_size), 
                                validation_data=(self.X_val_r, self.y_val_r), epochs=self.num_epochs, 
                                callbacks=[csv_logger, model_check])

        max_index=np.argmax(history.history['val_dice_coef'])
        max_dice_val=history.history['val_dice_coef'][max_index]
        dice_train=history.history['dice_coef'][max_index]
        logging.info('alpha= %s nfilter= %s blocks= %s node2_in= %s node3_in= %s node4_in= %s ops_node1= %s ops_node2= %s ops_node3= %s ops_node4= %s', 
                     str(self.gene[0]), str(self.gene[1]), str(self.gene[2]), str(self.gene[3]), str(self.gene[4]), 
                     str(self.gene[5]), str(self.gene[6]),str(self.gene[7]), str(self.gene[8]), str(self.gene[9]))

        # Save elapsed time
        elapsed = timeit.default_timer() - start_time
        logging.info('Time Elapsed: %s', str(elapsed))
        

