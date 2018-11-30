#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 03:33:03 2018

@author: n-kamiya
"""

import cv2

def otsu(input_tensor):
  
    gray = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2GRAY)
    

    ret, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)    


    cv2.imwrite("th2.jpg", th2)
    
    return th2


if __name__ == "__main__":
    otsu()