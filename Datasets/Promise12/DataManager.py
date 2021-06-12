#!/usr/bin/env python
# coding: utf-8

# In[3]:


import SimpleITK as sitk
import numpy as np
import os
import sys
from os import listdir
from os.path import isfile, join, splitext
import pandas as pd
from Datasets.Promise12.preprocessing import del_out_3D, norm_max_3D

#====================================================================
# This part of the code is based on V-Net
# from https://github.com/faustomilletari/VNet
# Milletari, F., Navab, N., & Ahmadi, S. A. (2016, October). V-net: Fully convolutional neural networks for volumetric medical image segmentation. In 2016 fourth international conference on 3D vision (3DV) (pp. 565-571). IEEE.
# ===================================================================


# In[14]:


class DataManager(object):
    def __init__(self,params):
        #path to dataset
        self.dirTrain=params["dirTrain"]
        # Voxel spacing
        self.dstRes = params["VolSpa"]
        # Fixed size of the image
        self.VolSize= params["VolSize"]
        # Total Number of images in the dataset
        self.NumImages=params["NumImages"]
        # Number of validation images
        self.TestImages=params["TestImages"]
        print("Preprocessing the training images...")

    def loadTrainingData(self):
        fileList=self.createImageFileList()
        gtList=self.createGTFileList(fileList)
        sitkImages, meanIntensityTrain=self.loadImages(fileList)
        sitkGT=self.loadGT(gtList)
        return sitkImages, sitkGT

    def createImageFileList(self):
        fileList = [f for f in listdir(self.dirTrain) if isfile(join(self.dirTrain, f)) and 'segmentation' not in f and '.mhd' in f]
        return fileList

    def createGTFileList(self, fileList):
        gtList=list()
        for f in fileList:
            filename, ext = splitext(f)
            gtList.append(join(filename + '_segmentation' + ext))
        return gtList

    def loadImages(self, fileList):
        sitkImages=dict()
        rescalFilt=sitk.RescaleIntensityImageFilter()
        rescalFilt.SetOutputMaximum(1)
        rescalFilt.SetOutputMinimum(0)
        stats = sitk.StatisticsImageFilter()
        m = 0.
        for f in fileList:
            sitkImages[f]=rescalFilt.Execute(sitk.Cast(sitk.ReadImage(join(self.dirTrain, f)),sitk.sitkFloat32))
            stats.Execute(sitkImages[f])
            m += stats.GetMean()
        meanIntensityTrain=m/len(sitkImages)
        return sitkImages, meanIntensityTrain

    def loadGT(self,gtList):
        sitkGT=dict()
        for f in gtList:
            sitkGT[f]=sitk.Cast(sitk.ReadImage(join(self.dirTrain, f))>0.5,sitk.sitkFloat32)
        return sitkGT

    def getNumpyImages(self,sitkImages,method):
        dat = self.getNumpyData(sitkImages,method)
        return dat

    def getNumpyData(self, dat,method):
        ret=dict()
        for key in dat:
            ret[key] = np.zeros([self.VolSize[0], self.VolSize[1], self.VolSize[2]], dtype=np.float32)
            img=dat[key]
            x_mm, y_mm, z_mm =img.GetSpacing()
            r, c, s= img.GetSize()
            factor = np.asarray(img.GetSpacing()) / [self.dstRes[0], self.dstRes[1], self.dstRes[2]]
            factorSize = np.asarray(img.GetSize() * factor, dtype=float)
            r_new, c_new, s_new =factorSize
            newSize = np.max([factorSize, self.VolSize], axis=0)
            newSize = newSize.astype(dtype=int).tolist()
            T=sitk.AffineTransform(3)
            T.SetMatrix(img.GetDirection())
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(img)
            resampler.SetOutputSpacing([self.dstRes[0], self.dstRes[1], self.dstRes[2]])
            resampler.SetSize(newSize)
            resampler.SetInterpolator(method)
            imgResampled = resampler.Execute(img)
            imgCentroid = np.asarray(newSize, dtype=float) / 2.0
            imgStartPx = (imgCentroid - self.VolSize / 2.0).astype(dtype=int)
            regionExtractor = sitk.RegionOfInterestImageFilter()
            regionExtractor.SetSize(self.VolSize.astype(dtype=int).tolist())
            regionExtractor.SetIndex(imgStartPx.tolist())
            imgResampledCropped = regionExtractor.Execute(imgResampled)
            ret[key] = np.transpose(sitk.GetArrayFromImage(imgResampledCropped).astype(dtype=float), [1, 2, 0])
        return ret

    def getNumpyGT(self,sitkGT, method):
        dat = self.getNumpyData(sitkGT,method)
        for key in dat:
            dat[key] = (dat[key]>0.5).astype(dtype=np.float32)
        return dat

    def save_matrix(self,list1, X, y, numpyImages,numpyGT):
        k=0
        for i in list1:
            if i<10:
                name_img="Case0"+str(i)+".mhd"
                name_seg="Case0"+str(i)+"_segmentation.mhd"
            else:
                name_img="Case"+str(i)+".mhd"
                name_seg="Case"+str(i)+"_segmentation.mhd"
            X[k,:,:,:,0]=numpyImages[name_img]
            y[k,:,:,:,0]=numpyGT[name_seg]
            k=k+1
        return X, y
    #Preprocessing images
    def pre_processing(self, X):
        X=del_out_3D(X, 3)
        X=norm_max_3D(X)
        return X

    def preprocess(self):
        # Load the images and groundtruth to dictionary
        sitkImages, sitkGT = self.loadTrainingData()

        # Transform from itk images to numpy images
        numpyImages = self.getNumpyImages(sitkImages,sitk.sitkLinear)
        numpyGT = self.getNumpyGT(sitkGT, sitk.sitkLinear)

        # Normalize images
        for key in numpyImages:
            mean = np.mean(numpyImages[key][numpyImages[key]>0])
            std = np.std(numpyImages[key][numpyImages[key]>0])
            numpyImages[key]-=mean
            numpyImages[key]/=std

        # Divide images into training and testing
        train_num=self.NumImages-self.TestImages
        ini=list(range(0,self.NumImages))
        testing=np.random.RandomState(seed=0).permutation(self.NumImages)

        # Selecting fold 1 division
        test1=np.sort(testing[0:10])
        train1=np.sort([e for e in ini if e not in test1])

        # Save images into matrices
        row,column,slices=self.VolSize
        X_train=np.zeros((train_num,row,column,slices,1))
        y_train=np.zeros((train_num,row,column,slices,1))
        X_test=np.zeros((self.TestImages,row,column,slices,1))
        y_test=np.zeros((self.TestImages,row,column,slices,1))

        self.X_train, self.y_train=self.save_matrix(train1, X_train, y_train, numpyImages,numpyGT)
        self.X_test, self.y_test=self.save_matrix(test1, X_test, y_test,numpyImages,numpyGT)

        # Normalize the data
        self.X_train= self.pre_processing(self.X_train)
        self.X_test= self.pre_processing(self.X_test)


        #Save as .npy matrices
        #np.save(self.dirProc+"X_train.npy",X_train)
        #np.save(self.dirProc+"y_train.npy",y_train)
        #np.save(self.dirProc+"X_test.npy",X_test)
        #np.save(self.dirProc+"y_test.npy",y_test)
