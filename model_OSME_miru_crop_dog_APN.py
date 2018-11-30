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
from otsu import otsu

#%%

K.clear_session()
BATCH_SIZE = 16
train_nb = 18571
num_classes = 120
img_size= 448
classes = []


train_path = "/home/n-kamiya/datasets/Standford_dogs_dataset/Images"

#%% create data generator 

train_datagen = ImageDataGenerator(rescale = 1./255, 
                                   zoom_range=[0.8,1],
                                   rotation_range=30,
                                   horizontal_flip=True,
                                   validation_split=0.1)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(img_size, img_size),
        batch_size=BATCH_SIZE,
        seed = 13,
        multi_outputs=True,
        subset="training",
        out_n=2
        )

validation_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(img_size, img_size),
        batch_size=BATCH_SIZE,
        seed = 13,
        multi_outputs=True,
        subset="validation",
        out_n=2
        )

#%% add random crop function
def random_crop(img, random_crop_size,seed):
    # Note: image_data_format is 'channel_last'
    np.random.seed(seed=seed)
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]


def crop_generator(batches, crop_length , seed):
    """Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator.
    """
    while True:
        batch_x, batch_y = next(batches)
        batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length),seed)
        yield (batch_crops, batch_y)

seed = 13
crop_size = 392
train_generator_cropped = crop_generator(train_generator, crop_size, seed)
validation_generator_cropped = crop_generator(validation_generator, crop_size, seed)
#%% finetuning resnet50

input_tensor = Input(shape=(crop_size, crop_size, 3))
#base_model = ResNet50(weights = "imagenet", include_top=False, input_tensor=input_tensor)
base_model = se_inception_v3.se_inception_v3(include_top=False, input_tensor=input_tensor)
#base_model.load_weights("/home/n-kamiya/models/model_without_MAMC/model_inceptv3_without_OSME.best_loss.hdf5",by_name=True)
base_model.load_weights("/home/n-kamiya/.keras/models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5", by_name=True)


for layer in base_model.layers:
    layer.trainable = False
#%%
#%%

channel_axis = 3

def osme_block(in_block, ch, ratio=16, name=None):
    z = GlobalAveragePooling2D()(in_block) # 1
    x = Dense(ch//ratio, activation='relu')(z) # 2
    x = Dense(ch, activation='sigmoid', name=name)(x) # 3
#    print(x.shape)
    return Multiply()([in_block,x]) # 4

attention = osme_block(base_model.output, base_model.output_shape[3], name='attention1')
output_1 = osme_block(base_model.output, base_model.output_shape[3], name='output_1')

#%%
crop_branch = output_1
output_1 = GlobalAveragePooling2D()(output_1)

attention = Convolution2D(1,(1,1), name='attetion_map', activation='sigmoid')(attention)
crop_branch = Convolution2D(1,(1,1), name='attetion_map_crop', activation='sigmoid')(crop_branch)

#%%
# TODO: implemetation of crop layer
def APN(input_list):
    # get input tensor
    input_tensor = input_list[0]
    
    global crop_size
    # get input_image tensor
    input_image = input_list[1]
    
    # assert it is a square image
    assert input_tensor.get_shape()[1] == input_tensor.get_shape()[2]
    # get length
    length = int(input_tensor.get_shape()[1])
    
    # create a zeros tensor for cropped image
#    cropped_imgs = K.zeros((0, int(crop_size/2), int(crop_size/2), 1))
    cropped_imgs = []
    
    tensor = K.reshape(input_tensor , shape=(tf.shape(input_tensor)[0], int(input_tensor.get_shape()[1])*(input_tensor.get_shape()[2]) ))
#    
#    max_nums = K.argmax(tensor[1])
    
    def condition(i, x):
    # Tensor("while/Merge:0", shape=(), dtype=int32) Tensor("while/Merge_1:0", shape=(), dtype=int32)
        return i < BATCH_SIZE

    def crop(i,x):

        max_value = K.argmax(x[i])

        _x = max_value % length
        _y = max_value / length
                
        x = tf.scalar_mul(int(crop_size/length), _x) # 2 * 3 
        y = tf.scalar_mul(int(crop_size/length), _y) # 2 * 3         
        
        x = tf.to_int32(x)
        y = tf.to_int32(y)
        
        l = crop_size/2
        margin = l/2 
        
        margin = K.constant(int(margin), dtype=tf.int32)
        crop_s = K.constant(int(crop_size), dtype=tf.int32)
        
        diff = tf.subtract(crop_s, margin)         
        
#        if x < margin or x > (crop_s-margin):
#            x = marginx =  
        cond_1 = tf.math.logical_or(tf.math.greater(x, diff),tf.math.less(x, margin))

#        
#        if y < margin or y > (crop_s-margin):
#            y = margin
        
        cond_2 = tf.math.logical_or(tf.math.less(y, margin),tf.math.greater(y, diff))

        x = tf.cond(cond_1, lambda: margin , lambda: x)
        y = tf.cond(cond_2, lambda: margin, lambda: y)

        
        cropped_imgs.append(input_image[:, x-margin:(x+margin), y-margin:(y+margin), :])
        
        i += 1
    
    tf.while_loop(crop, condition, (0, tensor))
    
    

    #return batch tensor
    return cropped_imgs 
    
#%%
#TODO: impementation of Lamda layer and shape
_input_tensor = Input(shape=(crop_size, crop_size, 3))

cropped_input = Lambda(APN)([crop_branch, _input_tensor])
    

 

#%%
#attention = otsu(attention)

x = Add()([base_model.output, attention])

# mixed 9: 8 x 8 x 2048
def last_block(x):
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)
    
        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = concatenate(
            [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))
    
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
            name='mixed' + str(9 + i))
    
        # squeeze and excite block
        x = squeeze_excite_block(x)
        return x

x = last_block(x)

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
checkpointer = ModelCheckpoint(filepath='/home/n-kamiya/models/model_without_MAMC/model_inceptv3_OSME_SE_miru_dogs_test1.best_loss.hdf5', verbose=1, save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                  patience=3, min_lr=0.000001)

#es_cb = EarlyStopping(patience=11)

#%% fit_generator
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size

history = model.fit_generator(train_generator_cropped,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=20,
                    validation_data=validation_generator_cropped,
                    validation_steps=STEP_SIZE_VALID,
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

with open('/home/n-kamiya/models/model_without_MAMC/history_inceptv3_with_OSME_miru_dogs_cropped{0:%d%m}-{0:%H%M%S}.json'.format(now), 'w') as f:
    json.dump(history.history, f,cls = MyEncoder)
    
    
#%% plot results

#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model_without_MAMC accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.savefig("/home/n-kamiya/models/model_without_MAMC/history_inceptv3_with_OSME_SE_miru_448{0:%d%m}-{0:%H%M%S}.png".format(now))
##plt.show()
#
##loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model_without_MAMC loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.savefig("/home/n-kamiya/models/model_without_MAMC/loss_inceptv3_without_OSME_SE_miru_448{0:%d%m}-{0:%H%M%S}.png".format(now))
##plt.show()