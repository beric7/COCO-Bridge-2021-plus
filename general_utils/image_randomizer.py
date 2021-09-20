# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:59:50 2020

@author: User
"""
import os
import shutil
import sys

BASE = 'E://'
sys.path.insert(0, BASE + 'Python/Scraping_Python_Pkg/')
from classification_utils import buildImageFileList

def sortRandomImages(src, dst, number):
    
    imageFilePaths, image_names = buildImageFileList(BASE, src, '')
    
    import random
    for i in range(0,number):
        randNumber = random.randrange(0, len(imageFilePaths), 1)
        print (randNumber)
        src_rand = imageFilePaths[randNumber]
        shutil.copy(src_rand, dst)
        imageFilePaths.remove(src_rand)
        