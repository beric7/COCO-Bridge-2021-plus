# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:16:02 2020

@author: Eric Bianchi
"""
import shutil

def copyfileByName(src, dst, filenames):
    for file in filenames:
        shutil.copy(src + file, dst)
    
def movefileByName(src, dst, filenames):
    for file in filenames:
        shutil.move(src + file, dst)
    
def removefileByName(src, dst, filenames):
    for file in filenames:
        shutil.remove(src + file, dst)