# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 12:55:35 2020

@author: Eric Bianchi
"""
import sys
import cv2
import os

sys.path.insert(0, 'D://Python/Scraping_Python_Pkg')
from readCSVfile import readFile

"""
@param: csv file = the csv file indicating the bounding box annotations on the parent images
@param: source image directory = where the parent images are stored
@param: destintation directory = where the sub-image will be saved to
"""
def csv_to_subimage(csv_file, sourceImageDirectory, destinationDirectory): 
    """This is a file that is specific to the current bounding box format that is produced 
    from the current XML to CSV conversion from the labelImg annotator."""
    
    rows, fields = readFile(csv_file)
    
    for rowList in rows:
        classType = rowList[3]
        
        try:
            destinationFolder = destinationDirectory + '/' + classType + '/'
            if not os.path.exists(destinationFolder):
                os.makedirs(destinationFolder)
            # end if
        except Exception as e:
            print("unable to create directory " + destinationFolder + "error: " + str(e))
        # end try
        dir_list = os.listdir(destinationFolder) 
        count = len(dir_list)
        
        image = cv2.imread(sourceImageDirectory + rowList[0])
        window = image[int(rowList[5]):int(rowList[7]),int(rowList[4]):int(rowList[6]):]
        destination = destinationDirectory + classType + '/' + classType
        
        cv2.imwrite(destinationFolder + str(count+1) + '_' + classType +'.jpg', window)
