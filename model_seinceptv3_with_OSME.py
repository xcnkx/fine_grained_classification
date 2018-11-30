#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 08:45:20 2018

@author: n-kamiya
"""
import keras
from keras.callbacks import EarlyStopping
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Multiply, add, concatenate
from keras.layers import GlobalAveragePooling2D, Dropout
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import SGD, RMSprop
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
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
import grad_cam
from keras.models import load_model
import os

import se_inception_v3
#%%
import tensorflow as tf
from keras.backend import tensorflow_backend

K.clear_session()

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

BATCH_SIZE = 16
test_nb = 5794
train_nb = 5994
num_classes = 200
img_size= 336
classes = []

train_path = "/home/n-kamiya/datasets/CUB2002011/CUB_200_2011/train/"
test_path = "/home/n-kamiya/datasets/CUB2002011/CUB_200_2011/test/"
#
#
with open("/home/n-kamiya/datasets/CUB2002011/CUB_200_2011/classes.txt") as f:
    for l in f.readlines():
        data = l.split()
        classes.append(data[1])

#%% create data generator 

train_datagen = ImageDataGenerator(rescale = 1.0/255,
#                                   featurewise_center=True,
#                                   featurewise_std_normalization=True,
                                   zoom_range=[0.7,1.0],
                                   rotation_range=30,
#                                   zca_whitening=True,
                                   horizontal_flip=True,
                                   vertical_flip=False,
                                   )
test_datagen = ImageDataGenerator(rescale = 1.0/255
        )


train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(img_size, img_size),
        batch_size=BATCH_SIZE,
        seed = 13,
        #save_to_dir='/home/n-kamiya/datasets/CUB2002011/CUB_200_2011/generated_images4/'
        )

validation_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(img_size, img_size),
        batch_size=BATCH_SIZE,
        seed = 13)

#%% finetuning 

input_tensor = Input(shape=(img_size, img_size, 3))
#base_model = VGG19(weights = "imagenet", include_top=False, input_tensor=input_tensor)
#base_model = ResNet50(weights = "imagenet", include_top=False, input_tensor=input_tensor)
base_model = se_inception_v3.se_inception_v3(weights = None, include_top=False, input_tensor=input_tensor)
base_model.load_weights("/home/n-kamiya/models/model_without_MAMC/model_osme_inceptv3_beta.best_loss.hdf5", by_name=True)

#for layer in base_model.layers:
#    layer.trainable = False
#        
#%% Implementation of OSME module

def osme_block(in_block, ch, ratio=16, name=None):
    z = GlobalAveragePooling2D()(in_block) # 1
    x = Dense(ch//ratio, activation='relu')(z) # 2
    x = Dense(ch, activation='sigmoid', name=name)(x) # 3
    return Multiply()([in_block, x]) # 4

s_1 = osme_block(base_model.output, base_model.output_shape[3], name='attention1')
s_2 = osme_block(base_model.output, base_model.output_shape[3], name='attention2')

fc1 = Flatten()(s_1)
fc2 = Flatten()(s_2)

fc1 = Dense(1024, name='fc1')(fc1)
fc2 = Dense(1024, name='fc2')(fc2)

#fc = fc1
fc = concatenate([fc1,fc2]) #fc1 + fc2

#fc = Dropout(0.5)(fc) #add dropout
prediction = Dense(num_classes, activation='softmax', name='prediction')(fc)


model = Model(inputs=base_model.input, outputs=prediction)

opt = SGD(lr=0.001, momentum=0.9, decay=0.0005)
#opt = RMSprop(lr=0.01)


#model.load_weights("/home/n-kamiya/models/model_without_MAMC/model_osme_inceptv3_beta.best_loss.hdf5", by_name=True)
    
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

#%%
plot_model(model, to_file="model.png", show_shapes=True)

#%% implement checkpointer and reduce_lr (to prevent overfitting)
checkpointer = ModelCheckpoint(filepath='/home/n-kamiya/models/model_without_MAMC/model_osme_se_inceptv3.best_loss.hdf5', verbose=1, save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                  patience=3, min_lr=0.0000001)

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
plt.savefig("/home/n-kamiya/models/model_without_MAMC/history_seinceptv3_with_OSME{0:%d%m}-{0:%H%M%S}.png".format(now))
plt.show()

#loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_without_MAMC loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("/home/n-kamiya/models/model_without_MAMC/loss_seinceptv3_gamma_with_OSME{0:%d%m}-{0:%H%M%S}.png".format(now))
plt.show()


#%%Visualize with grad_cam
#
model = load_model("/home/n-kamiya/models/model_without_MAMC/model_osme_se_inceptv3_gamma.best_loss_.hdf5", custom_objects={"tf":tf})
#%%
import copy
img_size = 336
for i, file in enumerate(os.listdir("/home/n-kamiya/datasets/CUB2002011/CUB_200_2011/test/013.Bobolink")):
    x = img_to_array(load_img(file, target_size=(img_size,img_size)))
    _x = copy.copy(x)    
    _x /= 255
    y_proba = model.predict(_x.reshape([-1,img_size,img_size,3]))
    print(classes[int(y_proba.argmax(axis = -1))])
    image1 = grad_cam.Grad_Cam(model, x, "multiply_1", img_size)
    image2 = grad_cam.Grad_Cam(model, x, "multiply_2", img_size)

    image1 = array_to_img(image1)
    image2 = array_to_img(image2)

    image1.save("/home/n-kamiya/images/inceptv3_alpha_with_OSME/%s_%d_attention1.png"%(file, i))
    image2.save("/home/n-kamiya/images/inceptv3_alpha_with_OSME/%s_%d_attention2.png"%(file, i))

model.evaluate_generator(generator=validation_generator, steps = test_nb/BATCH_SIZE, verbose = 1)
#%%
