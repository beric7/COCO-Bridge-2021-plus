# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:04:53 2020

@author: Eric Bianchi
"""
import os

def rename(direc, destination, typToSave, ID):           
    i = 1
     
    for filename in os.listdir(direc):
        
        # Rename images
        dst = ID + "_" +str(i) + "." + typToSave
        src = direc + '/'+ filename
        dst = destination + '/'+ dst
         
        # rename() function will
        # rename all the files
        os.rename(src, dst)
        i += 1


def DictionaryClassifier(path):
    
    classIdx = {}
    validationDict = {}
    classNumber = 0
    
    for folder in os.listdir(path):
      classIdx.update({folder:classNumber}) 
      
      for file in os.listdir(path + folder + '/'):
          validationDict.update({file:classNumber})
          
      classNumber = classNumber + 1
      
    return classIdx, validationDict




