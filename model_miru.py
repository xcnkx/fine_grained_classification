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
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Multiply, Lambda, concatenate, Add
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
import se_inception_v3_1 as se_inception_v3
from se_inception_v3 import conv2d_bn, squeeze_excite_block
import tensorflow as tf
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
        multi_outputs=True,
        )

validation_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(img_size, img_size),
        batch_size=BATCH_SIZE,
        seed = 13,
        multi_outputs=True,
        )
#%% finetuning resnet50

input_tensor = Input(shape=(img_size, img_size, 3))
#base_model = ResNet50(weights = "imagenet", include_top=False, input_tensor=input_tensor)
base_model = se_inception_v3.se_inception_v3(include_top=False, input_tensor=input_tensor)
base_model.load_weights("/home/n-kamiya/models/model_without_MAMC/model_inceptv3_without_OSME.best_loss.hdf5",by_name=True)
    

#for layer in base_model.layers:
#    layer.trainable = False
#%%
#%%

channel_axis = 3
# mixed 9: 8 x 8 x 2048


def last_block(x, layer):
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)
    
        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = concatenate(
            [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + layer + str(i))
    
        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)
    
        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i)+'_'+layer)
    
        # squeeze and excite block
        x = squeeze_excite_block(x)
        return x
    
#attention branch
def attention_block(in_block, ch, ratio=16, name=None):
    x = last_block(in_block,'attention')
    x = Convolution2D(num_classes,(1,1))(x)
    return x


    
x = attention_block(base_model.output, base_model.output_shape[3])
attention_map = Convolution2D(1,(1,1),activation='sigmoid', name="attention1")(x)
output_1 = Convolution2D(num_classes,(1,1))(x)
output_1 = GlobalAveragePooling2D(name="output1")(output_1)

x = Multiply()([base_model.output, attention_map])
x = Add()([base_model.output, x])

x = last_block(x,'perception')
output_2 = GlobalAveragePooling2D(name='output_2')(x)
 
attention_branch = Dense(num_classes, activation='softmax', name='attention_branch')(output_1)
perception_branch = Dense(num_classes, activation='softmax', name='perception_branch')(output_2)

#%%
## for these 2 outputs run it requires a custom datagenerator that return a tuple of yield like (x_batch, [y_batch, y_batch])
model = Model(inputs=base_model.input, outputs=[attention_branch,perception_branch])

opt = SGD(lr=0.01, momentum=0.9, decay=0.0005)

#model.load_weights("/home/n-kamiya/models/model_without_MAMC/model_inceptv3_without_OSME.best_loss.hdf5")
    
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

#%%
plot_model(model, to_file="model_inceptv3_miru.png", show_shapes=True)

#%% implement checkpointer and reduce_lr (to prevent overfitting)
checkpointer = ModelCheckpoint(filepath='/home/n-kamiya/models/model_without_MAMC/model_inceptv3_without_OSME_SE_miru_448.best_loss.hdf5', verbose=1, save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                  patience=3, min_lr=0.000001)

#es_cb = EarlyStopping(patience=11)

#%% fit_generator

history = model.fit_generator(train_generator,
                    steps_per_epoch=train_nb/BATCH_SIZE,
                    epochs=60,
                    validation_data=validation_generator,
                    validation_steps=64,
                    verbose=1,
                    callbacks=[reduce_lr, checkpointer])
#%%
        
import json
import datetime
now = datetime.datetime.now()

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

with open('history_inceptv3_with_OSME_SE_miru_448{0:%d%m}-{0:%H%M%S}.json'.format(now), 'w') as f:
    json.dump(history.history, f,cls = MyEncoder)
    
    
#%% plot results

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model_without_MAMC accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("/home/n-kamiya/models/model_without_MAMC/history_inceptv3_with_OSME_SE_miru_448{0:%d%m}-{0:%H%M%S}.png".format(now))
plt.show()

#loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_without_MAMC loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("/home/n-kamiya/models/model_without_MAMC/loss_inceptv3_without_OSME_SE_miru_448{0:%d%m}-{0:%H%M%S}.png".format(now))
plt.show()
