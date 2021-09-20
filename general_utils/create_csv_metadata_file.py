# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:23:16 2020

@author: Eric Bianchi
"""
import sys
import os
sys.path.insert(0, 'D://Python/Scraping_Python_Pkg')

from CSV_to_SUBimage import csv_to_subimage
from buildImageFileList import buildImageFileList
from writeToCSV import arrayToCSV

def makeInputSheet(folderName, fields): 
    imageFilePaths, image_names = buildImageFileList(folderName)
    brokenList=folderName.split('/')
    objectType = brokenList[len(brokenList)-2]
    destination = folderName + objectType+'.csv'
    arrayToCSV(image_names, fields, destination)

folderName = 'D://DATA/Report_Creation/subImage/Bearing/' 
# here you can add whatever to the list. 

#=============================================================================
# BEARING:
fields = ['Image Name', 'Bearing Class', '(1000) Corrosion', 
          '(1020) Connection', '(2210) Movement', '(2220) Alignment', 
          '(2230) Bulging, Splitting, or Tearing', '(2240) Loss of Bearing Area', 
          '(7000) Damage']

# GUSSET PLATE CONNECTION:
"""
fields = ['Image Name', '(1000) Corrosion', '(1010)' Cracking,
          '(1020) Connection', '(1900) Distortion', '(7000) Damage']
"""

# COVERPLATE TERMINATION:
"""
fields = ['Image Name', 'Overall']
"""
    
# OUT OF PLANE STIFFENER:
"""
fields = ['Image Name', 'Overall']
"""
#=============================================================================

# base folder path
base = 'D://DATA/Report_Creation/'

# csv_to_subimage(csvFile, source directory of images, destination of subimages)
# csv_to_subimage(base+'data.csv', base+'Test/', base+'subImage/')

# makeInputSheet(name of object folder, corresponding fields to include)
makeInputSheet(folderName, fields)