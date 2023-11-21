#!/usr/bin/env python
# coding: utf-8
# 
# Copyright 2022 Istituto Nazionale di Fisica Nucleare
# SPDX-License-Identifier: EUPL-1.2
#
# environment generation:
#
# conda create -n keras
# conda activate keras
# conda install tensorflow jupyter numpy matplotlib scikit-image
# conda install -c conda-forge nibabel
#
# note that conda-forge is used separately
# note that if you have a GPU you should install tensorflow-gpu and cudatoolkit
# in the first conda install line.

# # Lung Segmentation 
# 
# In this notebook we will implement a 3D convolutional U-NET 
# which is able to segment the lung region. We start by 
# importing the packages needed for this work.
#


import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from glob import glob
from tensorflow import keras
import tensorflow as tf


# We need to start by defining the data used for traning and 
# for testing. In this case, with only 60 images available, 
# we will divide the dataset into 2. 90% of the images will 
# be used in training while the remaining 10% will be used for 
# testing. To set the datasets follow the next steps:
# 
# * select the patients with glob
# * open the lung ct images with nibabel and store the windowed (1300,-350) 
#       image with values from 0 o 1 in a numpy array
# * open the lung segmentation images and store the data into a numpy array
# * Define the train dataset by taking the first 90% of data from both the preceding arrays
# * Define the test dataset by taking the last 10% of the data

def windower(data,wmin,wmax):
    """
    windower function, gets in input a numpy array,
    the minimum HU value and the maximum HU value.
    It returns a windowed numpy array with the same dimension of data
    containing values between 0 and 1
    """
    dump = data.copy()
    dump[dump>=wmax] = wmax
    dump[dump<=wmin] = wmin
    dump -= wmin
    w = wmax - wmin
    return dump / w

patients = glob("../data/lctsc/*/")

ct_images  = np.zeros((len(patients),64,64,64,1))
seg_images = np.zeros((len(patients),64,64,64,1))

for i,p in zip(range(len(patients)),patients):
    ct_images[i,...,0]  = windower(nib.load("{}ct.nii.gz".format(p)).get_fdata(),-1000,300)
    seg_images[i,...,0] = nib.load("{}lung.nii.gz".format(p)).get_fdata()
    seg_images[i,...,0][seg_images[i,...,0]>0] = 1
    
train_img = ct_images[0:int(0.9*len(patients))].astype(np.float32)
train_seg = seg_images[0:int(0.9*len(patients))].astype(np.float32)

test_img = ct_images[int(0.9*len(patients)):].astype(np.float32)
test_seg = seg_images[int(0.9*len(patients)):].astype(np.float32)

print(test_img.max(),test_img.min())
print(test_seg.max(),test_seg.min())


# Now we define the model

def model(input_shape=(64,64,64,1),k1=8,reg=0.1):
    inputs=keras.layers.Input(shape=(input_shape))
    
    initializer = keras.initializers.Constant(1.)
    params = {"kernel_regularizer" : keras.regularizers.l2(reg), 
              "kernel_initializer" : 'random_normal'}
    
    # down
    def down(previous_layer,k):
        d = keras.layers.Conv3D(k,3,padding="same",**params)(previous_layer)
        d = keras.layers.BatchNormalization()(d)
        d = keras.layers.Activation('relu')(d)
        d = keras.layers.Conv3D(k,3,padding="same",**params)(d)
        d = keras.layers.BatchNormalization()(d)
        d = keras.layers.Activation('relu')(d)
        d = keras.layers.Conv3D(k,3,padding="same",**params)(d)
        d = keras.layers.BatchNormalization()(d)
        d = keras.layers.MaxPooling3D()(d)

        res = keras.layers.Conv3D(k,3,strides=2,padding="same",**params)(previous_layer)
        d = keras.layers.Add()([res,d])
        d = keras.layers.Activation('relu')(d)
        return d
        
    d1 = down(inputs,k1)
    d2 = down(d1,k1*2)
    d3 = down(d2,k1*4)
    
    # bottleneck
    b = keras.layers.Conv3D(k1*8,3,padding="same",**params)(d3)
    b = keras.layers.BatchNormalization()(b)
    b = keras.layers.Activation('relu')(b)
    b = keras.layers.Conv3D(k1*8,3,padding="same",**params)(b)
    b = keras.layers.BatchNormalization()(b)
    b = keras.layers.Activation('relu')(b)
    
    # up
    def up(previous_layer,down_layer,k):
        u = keras.layers.Conv3DTranspose(k,3,padding="same",**params)(previous_layer)
        u = keras.layers.BatchNormalization()(u)
        u = keras.layers.Activation('relu')(u)
        u = keras.layers.Concatenate(axis=-1)([u,down_layer])
        u = keras.layers.Conv3D(k,3,padding="same",**params)(u)
        u = keras.layers.BatchNormalization()(u)
        u = keras.layers.Activation('relu')(u)
        u = keras.layers.Conv3D(k,3,padding="same",**params)(u)
        u = keras.layers.BatchNormalization()(u)
        u = keras.layers.UpSampling3D()(u)

        res = keras.layers.UpSampling3D()(previous_layer)
        res = keras.layers.Conv3D(k,1,padding="same",**params)(res)
        u = keras.layers.Add()([u,res])
        u = keras.layers.Activation('relu')(u)
        return u
        
    u1 = up(b ,d3,k1*4)
    u2 = up(u1,d2,k1*2)
    u3 = up(u2,d1,k1)
    
    out = keras.layers.Conv3D(1,3,padding="same",**params)(u3)
    out = keras.activations.sigmoid(out)
    
    model=keras.models.Model(inputs=inputs,outputs=out)
    model.summary(line_length=150)
    
    return model

keras.backend.clear_session()
unet = model()
unet.summary(line_length=150)

def DiceLoss(targets, inputs, smooth=1e-6):
    
    #flatten label and prediction tensors
    inputs = keras.backend.flatten(inputs)
    targets = keras.backend.flatten(targets)
    
    intersection = keras.backend.sum(targets*inputs)
    dice = (2*intersection + smooth) / (keras.backend.sum(targets) + keras.backend.sum(inputs) + smooth)
    return 1 - dice

def Dice(targets, inputs, smooth=1e-6):

    #flatten label and prediction tensors
    inputs = keras.backend.round(keras.backend.flatten(inputs))
    targets = keras.backend.round(keras.backend.flatten(targets))

    intersection = keras.backend.sum(targets*inputs)
    dice = (2*intersection + smooth) / (keras.backend.sum(targets) + keras.backend.sum(inputs) + smooth)
    return dice


adam = keras.optimizers.Adam(
    learning_rate=0.0001, 
    beta_1=0.9, 
    beta_2=0.999, 
    epsilon=1e-08, 
    amsgrad=True)

unet.compile(
    loss=DiceLoss,
    optimizer=adam,
    metrics=[Dice])


reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_Dice', 
    factor=0.5, patience=5, 
    min_lr=0.000005, verbose= True)

history = unet.fit(
    train_img,
    train_seg,
    epochs=30,
    batch_size=1,
    validation_data=(test_img,test_seg),
    callbacks=[reduce_lr])

#tf.keras.models.save_model(unet,'lung_unet')
unet.save('lung_unet.hdf5')

res = unet.predict(ct_images)

for i,r in zip(range(len(res)),res):
    plt.title("min: {:1.4f} - max: {:1.4f}".format(r.min(),r.max()))
    plt.imshow(r[...,32,0],vmin=0,vmax=1)
    plt.colorbar()
    plt.savefig("{}.png".format(i))
    plt.close()


