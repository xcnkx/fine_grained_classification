#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 08:45:20 2018

@author: n-kamiya
"""

from keras.callbacks import EarlyStopping
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Multiply
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import SGD, RMSprop
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
#%%

K.clear_session()
BATCH_SIZE = 16
test_nb = 5794
train_nb = 5994
num_classes = 200
img_size= 448
classes = []

train_path = "/home/n-kamiya/datasets/CUB2002011/CUB_200_2011/train/"
test_path = "/home/n-kamiya/datasets/CUB2002011/CUB_200_2011/test/"
#%%


with open("/home/n-kamiya/datasets/CUB2002011/CUB_200_2011/classes.txt") as f:
    for l in f.readlines():
        data = l.split()
        classes.append(data[1])


#%% create data generator 

train_datagen = ImageDataGenerator(rescale = 1./255, 
                                   zoom_range=[0.8,1],
                                   rotation_range=30,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(img_size, img_size),
        batch_size=BATCH_SIZE,
        seed = 13,
        )

validation_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(img_size, img_size),
        batch_size=BATCH_SIZE,
        seed = 13,
        )
#%% finetuning resnet50

input_tensor = Input(shape=(img_size, img_size, 3))
#base_model = ResNet50(weights = "imagenet", include_top=False, input_tensor=input_tensor)
base_model = InceptionV3(include_top=False, input_tensor=input_tensor)


#for layer in base_model.layers:
#    layer.trainable = False
    
#%%
# change only the output layer 
top_model = Sequential()
#top_model.add(Convolution2d(input_shape=base_model.output_shape[1:]))
top_model.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(1024,activation='relu'))
top_model.add(Dense(num_classes, activation='softmax'))

model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

#opt = SGD(lr=0.01, momentum=0.9, decay=0.0005)
opt = RMSprop(lr=0.01, momentum=0.9, decay=0.0005)

#model.load_weights("/home/n-kamiya/models/model_without_MAMC/model_inceptv3_without_OSME.best_loss.hdf5")
    
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

#%%
plot_model(model, to_file="model_inceptv3.png", show_shapes=True)

#%% implement checkpointer and reduce_lr (to prevent overfitting)
checkpointer = ModelCheckpoint(filepath='/home/n-kamiya/models/model_without_MAMC/model_inceptv3_without_OSME_SE.best_loss.hdf5', verbose=1, save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                  patience=3, min_lr=0.000001)

#es_cb = EarlyStopping(patience=11)

#%% fit_generator

history = model.fit_generator(train_generator,
                    steps_per_epoch=train_nb/BATCH_SIZE,
                    epochs=15,
                    validation_data=validation_generator,
                    validation_steps=64,
                    verbose=1,
                    callbacks=[reduce_lr, checkpointer])

#%% plot results
import datetime
now = datetime.datetime.now()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model_without_MAMC accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("/home/n-kamiya/models/model_without_MAMC/history_inceptv3_without_OSME_CAM{0:%d%m}-{0:%H%M%S}.png".format(now))
plt.show()

#loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_without_MAMC loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("/home/n-kamiya/models/model_without_MAMC/loss_inceptv3_without_OSME_CAM{0:%d%m}-{0:%H%M%S}.png".format(now))
plt.show()
