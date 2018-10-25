#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 08:25:38 2018

@author: n-kamiya
"""
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.regularizers import l2
import matplotlib.image as mpimg
import numpy as np
import keras.backend as K
import pathlib
from PIL import Image
import cv2
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import pandas as pd
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


#x_train = x_train.reshape(60000, 784)
#x_test = x_test.reshape(10000, 784)
img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first': # Theano backend
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:                                         # Tensorflow backend
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

batch_size = 128
num_classes = 10
epochs = 10


input_tensor = Input(shape=(img_rows, img_cols, 1))
base_model = ResNet50(weights = "imagenet", include_top=False, input_tensor=input_tensor)

# change only the output layer to a FC that it's output is a softmax layer
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(1024,activation="relu"))
top_model.add(Dropout(0.5))
top_model.add(Dense(num_classes, activation="softmax"))

model = Model(input=base_model.input, output=top_model(base_model.output))

for layer in base_model.layers:
    layer.trainable = False

opt = SGD(lr=0.01, momentum=0.9)


model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['top_3_categorical_accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs = epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])