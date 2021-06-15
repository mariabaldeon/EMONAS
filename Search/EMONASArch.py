
import numpy as np
from keras.models import Model, Sequential
from keras import initializers
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Add, Activation, Conv3DTranspose, Concatenate, Lambda
from keras import backend as K
import math
import tensorflow as tf
from keras.utils.vis_utils import plot_model

# Define the operations
OPS = {
    'conv3d_1x1x1': lambda filters, inp: conv3d_ReluConvBN(filters, (1,1,1), inp),
    'conv3d_3x3x3': lambda filters, inp: conv3d_ReluConvBN(filters, (3,3,3), inp),
    'conv3d_5x5x5': lambda filters, inp: conv3d_ReluConvBN(filters, (5,5,5), inp),
    'conv2d_3x3': lambda filters, inp: conv3d_ReluConvBN(filters, (3,3,1), inp),
    'conv2d_5x5': lambda filters, inp: conv3d_ReluConvBN(filters, (5,5,1), inp),
    'conv2d_7x7': lambda filters, inp: conv3d_ReluConvBN(filters, (7,7,1), inp),
    'convP3d_3x3': lambda filters, inp: convP3d_ReluConvBN(filters, (3,3,1), (1,1,3), inp),
    'convP3d_5x5': lambda filters, inp: convP3d_ReluConvBN(filters, (5,5,1), (1,1,5), inp),
    'convP3d_7x7': lambda filters, inp: convP3d_ReluConvBN(filters, (7,7,1), (1,1,7), inp),
    'identity': lambda filters, inp: Identity(filters, inp),
    }


# Define the operations for the stem convolution. No initial activation
StemOPS= {
    'conv3d_1x1x1': lambda filters, inp: conv3d_ConvBN(filters, (1,1,1), inp),
    'conv3d_3x3x3': lambda filters, inp: conv3d_ConvBN(filters, (3,3,3), inp),
    'conv3d_5x5x5': lambda filters, inp: conv3d_ConvBN(filters, (5,5,5), inp),
    'conv2d_3x3': lambda filters, inp: conv3d_ConvBN(filters, (3,3,1), inp),
    'conv2d_5x5': lambda filters, inp: conv3d_ConvBN(filters, (5,5,1), inp),
    'conv2d_7x7': lambda filters, inp: conv3d_ConvBN(filters, (7,7,1), inp),
    'convP3d_3x3': lambda filters, inp: convP3d_ConvBN(filters, (3,3,1), (1,1,3), inp),
    'convP3d_5x5': lambda filters, inp: convP3d_ConvBN(filters, (5,5,1), (1,1,5), inp),
    'convP3d_7x7': lambda filters, inp: convP3d_ConvBN(filters, (7,7,1), (1,1,7), inp),
    'identity': lambda filters, inp: Identity(filters, inp),
    }

def Identity(filters, inp):
    fil=inp.shape[-1]
    if fil==filters:
        return inp
    else:
        num_rep=filters//fil
        x=inp
        for i in range(num_rep-1):
            x=Concatenate()([inp,x])
        return x


def InNorm(x):
    return tf.contrib.layers.instance_norm(x)

def conv3d_ReluConvBN(filters, kernel, inp):
    x= Activation("relu")(inp)
    x= Conv3D(filters=filters, kernel_size=kernel, padding='same', kernel_initializer='he_uniform')(x)
    x= Lambda(InNorm)(x)
    return x
def conv3d_ConvBN(filters, kernel, inp):
    x= Conv3D(filters=filters, kernel_size=kernel, padding='same', kernel_initializer='he_uniform')(inp)
    x= Lambda(InNorm)(x)
    return x

# Pseudo-3D convolutions
def convP3d_ReluConvBN(filters, kernel1, kernel2, inp):
    x= Activation("relu")(inp)
    x= Conv3D(filters=filters, kernel_size=kernel1, padding='same', kernel_initializer='he_uniform')(x)
    x= Conv3D(filters=filters, kernel_size=kernel2, padding='same', kernel_initializer='he_uniform')(x)
    x= Lambda(InNorm)(x)
    return x
def convP3d_ConvBN(filters, kernel1, kernel2, inp):
    x= Conv3D(filters=filters, kernel_size=kernel1, padding='same', kernel_initializer='he_uniform')(inp)
    x= Conv3D(filters=filters, kernel_size=kernel2, padding='same', kernel_initializer='he_uniform')(x)
    x= Lambda(InNorm)(x)
    return x


def stem_cell(inp, ops_list, nfilter):
    x=StemOPS[ops_list[0]](filters=nfilter, inp=inp)
    return x

def frst_blck(inp,input_list, ops_list, nfilter ):
    node=[0]*5
    node[0]=inp
    node[1]=OPS[ops_list[0]](filters=nfilter, inp=node[0])
    node[2]=OPS[ops_list[1]](filters=nfilter, inp=node[input_list[0]])
    node[3]=OPS[ops_list[2]](filters=nfilter, inp=node[input_list[1]])
    node[4]=OPS[ops_list[3]](filters=nfilter, inp=node[input_list[2]])
    x=Add()([node[1], node[2], node[3], node[4]])
    return x

def downsampling_block(previous_block, input_list, ops_list, nfilter):
    node=[0]*5
    previous_block= MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(previous_block)
    node[0]=previous_block
    node[1]=OPS[ops_list[0]](filters=nfilter, inp=node[0])
    node[2]=OPS[ops_list[1]](filters=nfilter, inp=node[input_list[0]])
    node[3]=OPS[ops_list[2]](filters=nfilter, inp=node[input_list[1]])
    node[4]=OPS[ops_list[3]](filters=nfilter, inp=node[input_list[2]])
    x=Add()([node[1], node[2], node[3], node[4]])
    return x


def upsampling_block(downsampling_block, previous_block, input_list, ops_list, nfilter ):
    node=[0]*5
    previous_block=Conv3DTranspose(filters=nfilter, kernel_size=(2,2,2), strides=(2,2,2), padding='same',
                          kernel_initializer='he_uniform')(previous_block)
    x=Add()([previous_block, downsampling_block])
    node[0]=x
    node[1]=OPS[ops_list[0]](filters=nfilter, inp=node[0])
    node[2]=OPS[ops_list[1]](filters=nfilter, inp=node[input_list[0]])
    node[3]=OPS[ops_list[2]](filters=nfilter, inp=node[input_list[1]])
    node[4]=OPS[ops_list[3]](filters=nfilter, inp=node[input_list[2]])
    x=Add()([node[1], node[2], node[3], node[4]])
    return x


def get_EMONAS(input_list, ops_list, h, w, slices, channels, classes, nfilter, blocks):

    inp=Input((h, w, slices,channels))
    stem=stem_cell(inp, ops_list, nfilter)
    first_block=frst_blck(stem,input_list, ops_list, nfilter)

    if blocks==5:
        down1=downsampling_block(first_block, input_list, ops_list, nfilter*2)
        down2=downsampling_block(down1, input_list, ops_list, nfilter*4)
        up3=upsampling_block(down1, down2, input_list, ops_list, nfilter*2 )
        output= upsampling_block(first_block, up3, input_list, ops_list, nfilter)

    if blocks==7:
        down1=downsampling_block(first_block, input_list, ops_list, nfilter*2)
        down2=downsampling_block(down1, input_list, ops_list, nfilter*4)
        down3=downsampling_block(down2, input_list, ops_list, nfilter*8)
        up4=upsampling_block(down2, down3, input_list, ops_list, nfilter*4 )
        up5=upsampling_block(down1, up4, input_list, ops_list, nfilter*2)
        output= upsampling_block(first_block, up5, input_list, ops_list, nfilter)

    if blocks==9:
        down1=downsampling_block(first_block, input_list, ops_list, nfilter*2)
        down2=downsampling_block(down1, input_list, ops_list, nfilter*4)
        down3=downsampling_block(down2, input_list, ops_list, nfilter*8)
        down4=downsampling_block(down3, input_list, ops_list, nfilter*16)
        up5=upsampling_block(down3, down4, input_list, ops_list, nfilter*8 )
        up6=upsampling_block(down2, up5, input_list, ops_list, nfilter*4)
        up7=upsampling_block(down1, up6, input_list, ops_list, nfilter*2)
        output= upsampling_block(first_block, up7, input_list, ops_list, nfilter)

    output= Conv3D(filters=classes, kernel_size=(1,1,1), activation='sigmoid', kernel_initializer='he_uniform' )(output)
    model= Model(inputs=inp, outputs=output)
    return model

def prediction(kmodel, crpimg):
    imarr=np.array(crpimg).astype(np.float32)
    imarr = np.expand_dims(imarr, axis=0) 

    return kmodel.predict(imarr)
