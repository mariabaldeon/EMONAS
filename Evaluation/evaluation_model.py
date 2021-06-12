#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from keras import optimizers
from keras import backend as K
from Search.EMONASArch import get_EMONAS,prediction
from Train.model_train import dice_coef_loss
import math
import timeit
import os
import SimpleITK as sitk
import math


# In[ ]:


class ModelEvaluate(object): 
    def __init__(self,parameters, patch_size, pix_spacing, X_val, y_val):
        self.gene=parameters["gene"]
        self.stride=parameters["stride"]
        self.path=parameters["path"]
        self.patch_size=patch_size
        self.pix_spacing=pix_spacing
        self.X_test=X_val
        self.y_test=y_val
    
    # Computes dice, jaccard, falsenegative error, false positive and hausdorff dist
    def compute_overlap_measures(self,gt, img):
        overlap_measure_filter=sitk.LabelOverlapMeasuresImageFilter()
        hausdorff_distance_filter=sitk.HausdorffDistanceImageFilter()
        overlap_measure_filter.Execute(gt,img)
        dice=overlap_measure_filter.GetDiceCoefficient()
        jaccard=overlap_measure_filter.GetJaccardCoefficient()
        false_negative=overlap_measure_filter.GetFalseNegativeError()
        false_positive=overlap_measure_filter.GetFalsePositiveError()
        hausdorff_distance_filter.Execute(gt,img)
        haus=hausdorff_distance_filter.GetHausdorffDistance()
        return dice, jaccard, false_negative, false_positive, haus
    
    def compute_surface_dist(self, gt, img):

        reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(gt, squaredDistance=False,
                                                                       useImageSpacing=True))
        reference_surface = sitk.LabelContour(gt)
        statistics_image_filter = sitk.StatisticsImageFilter()
        statistics_image_filter.Execute(reference_surface)
        num_reference_surface_pixels = int(statistics_image_filter.GetSum())
        segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(img, squaredDistance=False,
                                                             useImageSpacing=True))
        segmented_surface = sitk.LabelContour(img)
        seg2ref_distance_map = reference_distance_map*sitk.Cast(segmented_surface, sitk.sitkFloat32)
        ref2seg_distance_map = segmented_distance_map*sitk.Cast(reference_surface, sitk.sitkFloat32)
        statistics_image_filter.Execute(segmented_surface)
        num_segmented_surface_pixels = int(statistics_image_filter.GetSum())
        seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
        seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr!=0])
        seg2ref_distances = seg2ref_distances +                         list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
        ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
        ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr!=0])
        ref2seg_distances = ref2seg_distances +                         list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))
        all_surface_distances = seg2ref_distances + ref2seg_distances
        mean_surface_distance = np.mean(all_surface_distances)
        median_surface_distance = np.median(all_surface_distances)
        std_surface_distance = np.std(all_surface_distances)
        per95_surface_distance = np.percentile(all_surface_distances, 95)
        return mean_surface_distance,median_surface_distance, std_surface_distance, per95_surface_distance
    
    def num_patches(self,img_dim, patch_dim, stride):
        n_patch=math.trunc(img_dim/stride)
        if img_dim%stride==0:
            total_patches=n_patch
            lst_idx=(n_patch-1)*stride
            end_patch=lst_idx+patch_dim
            padding=end_patch-img_dim
            return total_patches, padding
        lst_idx=n_patch*stride
        end_patch=lst_idx+patch_dim
        padding=end_patch-img_dim
        total_patches=n_patch+1
        return total_patches, padding
        
    def prediction_matrix_crop(self,X, model):
        num, row,col,sl, ch=X.shape
        pt_row, pt_col, pt_sl=self.patch_size
        str_row, str_col, str_sl=self.stride

        num_row, pad_row=self.num_patches(row, pt_row, str_row)
        num_col, pad_col=self.num_patches(col, pt_col, str_col)
        num_sl, pad_sl=self.num_patches(sl, pt_sl, str_sl)
        X_pad=np.zeros((num, row+pad_row, col+pad_col, sl+pad_sl, ch))
        X_pad[:, pad_row:, pad_col:, pad_sl:,:]=X
        y_pred_matrix=np.zeros(X_pad.shape)

        V=np.zeros(X_pad.shape)
        for i in range(num):
            for j in range(num_row):
                for k in range(num_col):
                    for m in range(num_sl):
                        row_in=j*str_row
                        col_in=k*str_col
                        sl_in=m*str_sl
                        row_fin=row_in+pt_row
                        col_fin=col_in+pt_col
                        sl_fin=sl_in+pt_sl
                        Xi=X_pad[i,row_in:row_fin,col_in:col_fin,sl_in:sl_fin,:]
                        yi=prediction(model, Xi) 
                        y_pred_matrix[i,row_in:row_fin,col_in:col_fin,sl_in:sl_fin,:]=y_pred_matrix[i,row_in:row_fin,col_in:col_fin,sl_in:sl_fin,:]+yi
                        Vi=np.zeros(X_pad.shape)
                        Vi[i,row_in:row_fin,col_in:col_fin,sl_in:sl_fin,:]=1.
                        V=V+Vi
        #compute the average of the predictions
        y_pred_matrix=np.true_divide(y_pred_matrix, V)
        y_pred_matrix=y_pred_matrix[:, pad_row:, pad_col:, pad_sl:,:]
        return y_pred_matrix
        
    def connected_component(self, y_pred):
        num, r, c, s, ch= y_pred.shape
        y_new=np.zeros(y_pred.shape)
        for i in range(num):
            yi=y_pred[i,:,:,:,0]
            yi=sitk.GetImageFromArray(yi)
            thfilter=sitk.BinaryThresholdImageFilter()
            thfilter.SetInsideValue(1)
            thfilter.SetOutsideValue(0)
            thfilter.SetLowerThreshold(0.5)
            yi = thfilter.Execute(yi)
            cc = sitk.ConnectedComponentImageFilter()
            yi = cc.Execute(sitk.Cast(yi,sitk.sitkUInt8))
            arrCC=np.transpose(sitk.GetArrayFromImage(yi).astype(dtype=float), [1, 2, 0])
            lab=np.zeros(int(np.max(arrCC)+1),dtype=float)
            for j in range(1,int(np.max(arrCC)+1)):
                lab[j]=np.sum(arrCC==j)
            activeLab=np.argmax(lab)
            yi = (yi==activeLab)
            yi=sitk.GetArrayFromImage(yi).astype(dtype=float)
            y_new[i,:,:,:,0]=yi
        return y_new
    
    # Evaluates 11 metrics of overlap and distance measure
    def eval_metrics_dist(self,y_true, y_pred, set_spacing, x,y,z):
        thres=0.5
        tpy="uint8"
        num, h, w, s, c=y_true.shape
        y_true=y_true.reshape(y_true.shape[:-1])
        y_true=y_true.astype(dtype=tpy)
        y_pred=np.where(y_pred>=thres,1,0)
        y_pred=y_pred.reshape(y_pred.shape[:-1])
        y_pred=y_pred.astype(dtype=tpy)

        metric_coef_ind=np.zeros((num,10))

        for i in range(num):
            y_predi=y_pred[i,:,:,:]
            y_truei=y_true[i,:,:,:]
            img=sitk.GetImageFromArray(y_predi)
            gt=sitk.GetImageFromArray(y_truei)
            if set_spacing:
                img.SetSpacing((x[i], y[i], z[i]))
                gt.SetSpacing((x[i], y[i], z[i]))

            # If there is no segmentation in the groundtruth or segmentation
            # cannot compute the sensitivity, dice coeff, jaccard and surface distance measurements
            if (len(np.unique(y_truei))==1 or len(np.unique(y_predi))==1):
                metric_coef_ind[i,0]=np.nan
                metric_coef_ind[i,1]=np.nan
                metric_coef_ind[i,2]=np.nan
                metric_coef_ind[i,3]=np.nan
                metric_coef_ind[i,4]=np.nan
                metric_coef_ind[i,5]=np.nan
                metric_coef_ind[i,6]=np.nan
                metric_coef_ind[i,7]=np.nan
                metric_coef_ind[i,8]=np.nan
                metric_coef_ind[i,9]=np.nan

            # If there is segmentation in the groundtruth
            # Compute the sensitivity, dice coeff, jaccard and surface distance measurements
            if (len(np.unique(y_truei))==2 and len(np.unique(y_predi))==2):
                dice, jaccard, false_negative, false_positive, haus=self.compute_overlap_measures(gt, img)
                mean_surface_distance,median_surface_distance, std_surface_distance, per95_surface_distance=self.compute_surface_dist(gt, img)
                metric_coef_ind[i,0]=dice
                metric_coef_ind[i,1]=1-false_positive
                metric_coef_ind[i,2]=jaccard
                metric_coef_ind[i,3]=false_negative
                metric_coef_ind[i,4]=false_positive
                metric_coef_ind[i,5]=haus
                metric_coef_ind[i,6]=mean_surface_distance
                metric_coef_ind[i,7]=median_surface_distance
                metric_coef_ind[i,8]=std_surface_distance
                metric_coef_ind[i,9]=per95_surface_distance
            metrics_cnn=pd.DataFrame(metric_coef_ind, columns=["val_dice", "val_recall","val_jaccard","val_false_negative_error", "val_false_positive_error", "val_haussdorf_dist","val_mean_surface_distance", "val_median_surface_distance","val_std_surface_distance", "val_95_haussdorf_dist"])
        return metrics_cnn
    
    def evaluate(self):
        
        weights=[]
        for f in os.listdir(self.path):
            if ".hdf5" in f:
                weights.append(os.path.join(self.path,f))
        
        # Training logs 
        location="EvalLogs"
        if not os.path.exists(location):
            os.makedirs(location)
        
        num,_,_,_,channels=self.X_test.shape
        classes=self.y_test.shape[-1]
        
        x=[self.pix_spacing[0]]*num
        y=[self.pix_spacing[1]]*num
        z=[self.pix_spacing[2]]*num
        
        #Use of connected component
        cc=True
        CompiledMetrics = pd.DataFrame(columns=['weight','val_dice','val_haus',
                                           'val_MSD','val_recall'])
        for i in range(len(weights)):

            input_list=[self.gene[3],self.gene[4],self.gene[5]]
            op_list=[self.gene[6],self.gene[7],self.gene[8],self.gene[9]]
            model=get_EMONAS(input_list, op_list, h=self.patch_size[0], w=self.patch_size[1],
                       slices=self.patch_size[2], channels=channels, classes=classes, nfilter=self.gene[1],blocks=self.gene[2])

            model.load_weights(weights[i])
            model.summary()
            adam=optimizers.Adam(lr=self.gene[1], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            model.compile(loss=dice_coef_loss, optimizer=adam, metrics=['accuracy'])

            y_test_pred=self.prediction_matrix_crop(self.X_test, model)

            # Apply Connected Component analysis
            if cc:
                y_test_pred=self.connected_component(y_test_pred)

            metrics_test =self.eval_metrics_dist(self.y_test, y_test_pred, True, x,y,z)

            # Save Metrics
            metrics_save =metrics_test[["val_dice", "val_recall","val_mean_surface_distance", "val_95_haussdorf_dist"]].copy()
            metrics_save.to_csv(str(location)+"/"+str(i)+"MetricsVal.csv")

            CompiledMetrics = CompiledMetrics.append({'weight':str(weights[i]),'val_dice':str(metrics_test['val_dice'].mean()),
                                                'val_haus':str(metrics_test["val_95_haussdorf_dist"].mean()),
                                                'val_MSD':str(metrics_test["val_mean_surface_distance"].mean()),
                                                'val_recall':str(metrics_test["val_recall"].mean())}, ignore_index=True)


            CompiledMetrics.to_csv(str(location)+'/CompiledMetrics.csv')

