# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 14:08:18 2020

@author: Eric Bianchi
"""

import numpy as np
import cv2
import os

def imageGrid(image, stepSize):
    """
    Parameters
    ----------
    image : numpy or cv2 array
        This must already be in the image format
    stepSize : int
        The step size to split the image

    Returns
    -------
    grid : array of images
        array of images for saving. 

    """
    grid = []
    for x in range(0, image.shape[1], stepSize):
        for y in range(0, image.shape[0], stepSize):
            window = image[y:y + stepSize, x:x + stepSize:]
            grid.append(window)
    return grid


def split_and_save_image(image_path, file_name, extension, destination_folder, stepSize):
    """
    Parameters
    ----------
    image_path : string
        string path to image file.
    file_name : string
        filename without extension
    extension : string
        extension to save the image as.
    destination_folder : string
        destination folder to save the file.
    stepSize : int
        The step size to split the image.

    Returns
    -------
    None.

    """
    if not os.path.exists(destination_folder): # if it doesn't exist already
        os.makedirs(destination_folder)
    
    
    image = cv2.imread(image_path)
    grid = imageGrid(image, stepSize)
    
    count = 0
    for img in grid:
        cv2.imwrite(destination_folder + file_name + '_' + str(count) + '.' + extension, img)
        count = count + 1
        

 



    
    
