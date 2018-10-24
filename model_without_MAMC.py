#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 08:45:20 2018

@author: n-kamiya
"""

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
import matplotlib.image as mpimg
import numpy as np
import keras.backend as K
import pathlib
from PIL import Image
import cv2


K.clear_session()

num_classes = 200
img_size=488
image_txt = "/home/n-kamiya/datasets/CUB2002011/CUB_200_2011"
images_path = "/home/n-kamiya/datasets/CUB2002011/CUB_200_2011/images"
        
#%%        
#def load_images(root):
#    all_imgs = []
#    all_classes = []
#    img_path = []
#    Xtrain = []
#    ytrain = []
#    Xtest = []
#    ytest = []
#    s = []
#    
#    with open("%s/images.txt" % root) as f:
#        for line in f.readlines():
#            data = line.split()
#            img_path.append(data[1])
#            class_name = data[1].split(".")
#            all_classes.append(class_name[0])
#            
#    with open("%s/train_test_split.txt" % root) as f:
#        for line in f.readlines():
#                data = line.split()
#                s.append(data)
#                             
#    for image in img_path:
#        path = images_path +'/'+ image
#        img = cv2.imread(path)
#        resize_img_ar = cv2.resize(img,(img_size, img_size))
#        all_imgs.append(resize_img_ar)
#          
#    for img in s:
#        if img[1] == "1":
#            Xtrain.append(all_imgs[int(img[0])-1])
#            ytrain.append(all_classes[int(img[0])-1])
#        elif img[1] == "0":
#            Xtest.append(all_imgs[int(img[0])-1])
#            ytest.append(all_classes[int(img[0])-1])
#        
#        
#        
#    return np.array(Xtrain), np.array(ytrain), np.array(Xtest), np.array(ytest)

#%%
#X_train, y_train, X_test, y_test = load_images(image_txt)
#%%
train_datagen = ImageDataGenerator(rescale=1./img_size)

test_datagen = ImageDataGenerator(rescale=1./img_size)

train_generator = train_datagen.flow_from_directory(images_path ,X_train, y_train, batch_size=64, seed = 13)
validation_generator = train_datagen.flow_from_directory(images_path ,X_train, y_train, batch_size=64, seed = 13)
#%%
base_model = ResNet50(weights = "imagenet", include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(1, kernel_initializer="glorot_uniform", activation="sigmoid", kernel_regularizer=l2(.0005))(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

opt = SGD(lr=.01, momentum=.9)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
#%%
checkpointer = ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                  patience=5, min_lr=0.001)

history = model.fit_generator(train_generator,
                    steps_per_epoch=2000,
                    epochs=10,
                    validation_data=test_generator,
                    validation_steps=800,
                    verbose=1,
                    callbacks=[reduce_lr, checkpointer])
