#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 08:45:20 2018

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



K.clear_session()

num_classes = 200
img_size= 255
image_txt = "/home/n-kamiya/datasets/CUB2002011/CUB_200_2011"
images_path = "/home/n-kamiya/datasets/CUB2002011/CUB_200_2011/images"
        
#%%        
def load_images(root):
    all_imgs = []
    all_classes = []
    img_path = []
    Xtrain = []
    ytrain = []
    Xtest = []
    ytest = []
    s = []
    
    #Get classes names from images.txt 
    with open("%s/images.txt" % root) as f:
        for line in f.readlines():
            data = line.split()
            img_path.append(data[1])
            class_name = data[1].split(".")
            class_name = class_name[1].split("/")
            all_classes.append(class_name[0])
    #Get train/test data         
    with open("%s/train_test_split.txt" % root) as f:
        for line in f.readlines():
                data = line.split()
                s.append(data)
    #run a loop to get all images formating to array                          
    for image in img_path:
        path = images_path +'/'+ image
        resize_img_ar = img_to_array(load_img(path, target_size=(img_size, img_size)))
        all_imgs.append(resize_img_ar)
    #split train/test data      
    for img in s:
        if img[1] == "1":
            Xtrain.append(all_imgs[int(img[0])-1])
            ytrain.append(all_classes[int(img[0])-1])
        elif img[1] == "0":
            Xtest.append(all_imgs[int(img[0])-1])
            ytest.append(all_classes[int(img[0])-1])
        
    return np.array(Xtrain), np.array(ytrain), np.array(Xtest), np.array(ytest)

#%% Loading image files
X_train, y_train, X_test, y_test = load_images(image_txt)
#%% Labeling y data
y_train_list = list(y_train.tolist())
y_test_list = list(y_test.tolist())
le_train = LabelEncoder()
le_test = LabelEncoder()
y_train_ar = le_train.fit_transform(y_train_list)
y_test_ar = le_test.fit_transform(y_test_list)

y_train = np.array(y_train_ar)
y_test = np.array(y_test_ar)

#%% one_hot encoder with keras to_categorical
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

#%% create data generator 
train_datagen = ImageDataGenerator(rescale = 1./img_size)
test_datagen = ImageDataGenerator(rescale = 1./img_size)

train_generator = train_datagen.flow(X_train, y_train, batch_size=32, seed = 13)
validation_generator = train_datagen.flow(X_train, y_train, batch_size=32, seed = 13)
#%% finetuning resnet50

input_tensor = Input(shape=(img_size, img_size, 3))
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

opt = SGD(lr=.01, momentum=.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

#%%
plot_model(model, to_file="model.png", show_shapes=True)

#%% implement checkpointer and reduce_lr (to prevent overfitting)
checkpointer = ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                  patience=5, min_lr=0.001)

#
history = model.fit_generator(train_generator,
                    steps_per_epoch=len(X_test),
                    epochs=10,
                    validation_data=validation_generator,
                    validation_steps=len(X_train),
                    verbose=1,
                    callbacks=[reduce_lr, checkpointer])

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()