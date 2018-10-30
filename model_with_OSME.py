#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 08:45:20 2018

@author: n-kamiya
"""

from keras.callbacks import EarlyStopping
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Multiply, add
from keras.layers import GlobalAveragePooling2D
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
                                   zoom_range=[0.5,1],
                                   rotation_range=90,
                                   horizontal_flip=True,
                                   vertical_flip=True)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(img_size, img_size),
        batch_size=BATCH_SIZE,
        seed = 13)

validation_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(img_size, img_size),
        batch_size=BATCH_SIZE,
        seed = 13)
#%% finetuning resnet50

input_tensor = Input(shape=(img_size, img_size, 3))
base_model = VGG19(weights = "imagenet", include_top=False, input_tensor=input_tensor)
#base_model = ResNet50(weights = "imagenet", include_top=False, input_tensor=input_tensor)

for layer in base_model.layers:
    layer.trainable = False
    
#%% Implementation of OSME module

def osme_block(in_block, ch, ratio=16):
    z = GlobalAveragePooling2D()(in_block) # 1
    x = Dense(ch//ratio, activation='relu')(z) # 2
    x = Dense(ch, activation='sigmoid')(x) # 3
    return Multiply()([in_block, x]) # 4

s_1 = osme_block(base_model.output, base_model.output_shape[3])
s_2 = osme_block(base_model.output, base_model.output_shape[3])

fc1 = Dense(1024, name='fc1')(s_1)
fc2 = Dense(1024, name='fc2')(s_2)

fc1 = Flatten()(fc1)
fc2 = Flatten()(fc2)

fc = add([fc1,fc2]) # fc1 + fc2

prediction = Dense(num_classes, activation='softmax', name='prediction')(fc)


model = Model(inputs=base_model.input, outputs=prediction)

opt = SGD(lr=0.001, momentum=0.9)

#model.load_weights("/home/n-kamiya/models/model_without_MAMC/model_osme.best_loss.hdf5")
    
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

#%%
plot_model(model, to_file="model.png", show_shapes=True)

#%% implement checkpointer and reduce_lr (to prevent overfitting)
checkpointer = ModelCheckpoint(filepath='/home/n-kamiya/models/model_without_MAMC/model_osme_no_weights.best_loss.hdf5', verbose=1, save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                  patience=3, min_lr=0.000001)

es_cb = EarlyStopping(patience=5)

#%% fit_generator

history = model.fit_generator(train_generator,
                    steps_per_epoch=train_nb/BATCH_SIZE,
                    epochs=60,
                    validation_data=validation_generator,
                    validation_steps=64,
                    verbose=1,
                    callbacks=[es_cb, reduce_lr, checkpointer])

#%% plot results
import datetime
now = datetime.datetime.now()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model_without_MAMC accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("/home/n-kamiya/models/model_without_MAMC/history_osme_{0:%d%m}-{0:%H%M%S}.png".format(now))
plt.show()

#loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_without_MAMC loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("/home/n-kamiya/models/model_without_MAMC/loss_osme_{0:%d%m}-{0:%H%M%S}.png".format(now))
plt.show()
