# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:04:53 2020

@author: Eric Bianchi
"""
import cv2
import numpy as np

def imShow_components(labels):
    label_hue = np.uint8(179*labels/labels)
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    
    labeled_img[label_hue==0] = 0
    
    #cv2.imshow("lableled.png", labeled_img)
    #cv2.imwrite(imgFolder + "Labeled.png", labeled_img)
    return labeled_img

def crackDetect(im):
    
    #imgFile = "test2.jpg"
    kernelSize = 41
    sigma = 20
    
    colorIm = im
    
    image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    image2 = cv2.GaussianBlur(image,(kernelSize,kernelSize), sigma)
    
    imageNew = cv2.subtract(image2, image)
    #cv2.imshow("def", imageNew) # Show smoothed image
    #cv2.imwrite(imgFolder + "Subtracted.png", imageNew)
    
    watershed = cv2.threshold(imageNew, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #cv2.imshow("watershed", watershed) # Show watershed image
    #cv2.imwrite(imgFolder + "WaterShed.png", watershed)
    
    #thr, binary = cv2.thershold(255-imageNew, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, labels = cv2.connectedComponents(watershed)
    
    segmentedIm = imShow_components(labels)
    
    superimposedIm = cv2.add(colorIm, segmentedIm)
    
    return superimposedIm
