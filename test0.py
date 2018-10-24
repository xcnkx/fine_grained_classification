#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 06:26:57 2018

@author: n-kamiya
"""

from keras.models import Model
from keras.layers import Input, Activation, merge, Dense, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2DTranspose, Cropping2D
from keras.layers.normalization import BatchNormalization



def create_model():

    FCN_CLASSES = 21

    #(samples, channels, rows, cols)
    input_img = Input(shape=(3, 224, 224))
    #(3*224*224)
    x = Convolution2D(64, 3, 3, activation='relu',padding='same')(input_img)
    x = Convolution2D(64, 3, 3, activation='relu',padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #(64*112*112)
    x = Convolution2D(128, 3, 3, activation='relu',padding='same')(x)
    x = Convolution2D(128, 3, 3, activation='relu',padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #(128*56*56)
    x = Convolution2D(256, 3, 3, activation='relu',padding='same')(x)
    x = Convolution2D(256, 3, 3, activation='relu',padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #(256*56*56)

    #split layer
    p3 = x
    p3 = Convolution2D(FCN_CLASSES, 1, 1,activation='relu')(p3)
    #(21*28*28)

    x = Convolution2D(512, 3, 3, activation='relu',padding='same')(x)
    x = Convolution2D(512, 3, 3, activation='relu',padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #(512*14*14)

    #split layer
    p4 = x
    p4 = Convolution2D(FCN_CLASSES, 1, 1, activation='relu')(p4)
    p4 = Conv2DTranspose(FCN_CLASSES, 4, 4,
            output_shape=(None, FCN_CLASSES, 30, 30),
            subsample=(2, 2),
            padding='valid')(p4)
    p4 = Cropping2D(cropping=((1, 1), (1, 1)))(p4)

    #(21*28*28)

    x = Convolution2D(512, 3, 3, activation='relu',padding='same')(x)
    x = Convolution2D(512, 3, 3, activation='relu',padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #(512*7*7)

    p5 = x
    p5 = Convolution2D(FCN_CLASSES, 1, 1, activation='relu')(p5)
    p5 = Conv2DTranspose(FCN_CLASSES, 8, 8,
            output_shape=(None, FCN_CLASSES, 32, 32),
            subsample=(4, 4),
            padding='valid')(p5)
    p5 = Cropping2D(cropping=((2, 2), (2, 2)))(p5)
    #(21*28*28)

    # merge scores
    merged = merge([p3, p4, p5], mode='sum')
    x = Conv2DTranspose(FCN_CLASSES, 16, 16,
            output_shape=(None, FCN_CLASSES, 232, 232),
            subsample=(8, 8),
            padding='valid')(merged)
    x = Cropping2D(cropping=((4, 4), (4, 4)))(x)
    x = Reshape((FCN_CLASSES,224*224))(x)
    x = Permute((2,1))(x)
    out = Activation("softmax")(x)
    #(21,224,224)
    model = Model(input_img, out)
    return model