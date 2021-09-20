# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:44:50 2020

@author: Eric Bianchi
CSV to Image
"""

import sys
import cv2
import os

"""
@param: csv file = the csv file indicating the bounding box annotations on the parent images
@param: source image directory = where the parent images are stored
@param: destintation directory = where the sub-image will be saved to
"""
def csv_to_image(csv_file, sourceImageDirectory, destinationDirectory, module_path, className=''): 
    """This is a file that is specific to the current bounding box format that is produced 
    from the current XML to CSV conversion from the labelImg annotator."""
    
    sys.path.insert(0, module_path)
    from readCSVfile import readFile
    
    rows, fields = readFile(csv_file)
    
    for rowList in rows:
        classType = className
        
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
        destination = destinationDirectory + classType + '/' + classType
        
        cv2.imwrite(destinationFolder + str(count+1) + '_' + classType +'.jpg', image)