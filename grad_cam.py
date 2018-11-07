#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 05:48:40 2018

@author: n-kamiya
"""

import pandas as pd
import numpy as np
import cv2
from keras import backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import load_model

def Grad_Cam(input_model, x, layer_name):
    '''
    Args:
       input_model: model object
       x: image(array)
       layer_name: convolution layer's name

    Returns:
       jetcam: heat map image(array)

    '''
    img_size = 336
    # preprocessing
    X = np.expand_dims(x, axis=0)

    X = X.astype('float32')
    preprocessed_input = X / 255.0
    
    model = input_model

    # predict class

    predictions = model.predict(preprocessed_input)
    class_idx = np.argmax(predictions[0])
    class_output = model.output[:, class_idx]


    # get gradients

    conv_output = model.get_layer(layer_name).output   # layer_name's output
    grads = K.gradients(class_output, conv_output)[0]  # gradients(loss, variables) 
    gradient_function = K.function([model.input], [conv_output, grads])

    output, grads_val = gradient_function([preprocessed_input])
    output, grads_val = output[0], grads_val[0]

    # get mean of weights, multiply with layer output
    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)


    # convert to image and combine with heatmap

    cam = cv2.resize(cam, (img_size, img_size), cv2.INTER_LINEAR) 
    cam = cam / cam.max()

    jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  
    jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)  
    jetcam = (np.float32(jetcam) + x / 2)

    return jetcam

#if __name__ == '__main__':
#
#    model = load_model("/home/n-kamiya/models/model_without_MAMC/model_inceptv3_without_OSME.best_loss.hdf5")
#    x = img_to_array(load_img('/home/n-kamiya/datasets/CUB2002011/CUB_200_2011/images/020.Yellow_breasted_Chat/Yellow_Breasted_Chat_0039_21654.jpg', target_size=(448,448)))
#    newx= np.expand_dims(x,axis=0)
#    preds = model.predict(newx)
#    print(classes[preds.argmax()])