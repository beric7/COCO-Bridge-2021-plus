# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 23:35:32 2020

DESCRIPTION:
Run all files at the same time:
    - convert xml to csv
    - augment and resize as desired
    - recordBbox changes

@author: Eric Bianchi
"""
#import argparse
module_path = "E://Python/4_augment_data/"

xml_source = './Train/original/XML_and_Images/'
csvFileDestination = './Train/original/bbox/csv/train_labels.csv'
CSV_save_file_location = './Train/320x320/bbox/csv/'
typeToSave = "jpeg" 
imageDatasetFolder = './Train/original/Images/'
temporaryFolder = './Train/original/temp/'
saveToFolder = './Train/320x320/Images/'
aug_num = 1
augmentType = "normal"
rescale = True
square = True 
dimension = 320

import sys
sys.path.insert(0, module_path) 


def rescaleImages():
    # args = args[0]
    # import sys
    from xml_to_csv import main as xml_to_csv
    from augment_and_resize_images import augAndResize
    from image_utils import rescale
    from record_bbox_info import recordBbox
    
    # =========================================================================    
    """
    xmlSource = args.xmlSource
    csvFileDestination = args.csvFileDestination
    """
    print('==================================================')
    print('Gathering XML and converting to CSV...')
    # xml_to_csv(xml_source, csvFileDestination)
    # =========================================================================
    
    # =========================================================================
    # Augment and re-size images
    """
    @param: typToSave = jpg, jpeg, png, etc. [string]
    @param: imageDataset = folder of images [path]
    @param: copiedDirectory = folder to save images temporarily [path]
    @param: SaveToFolder = folder where finished images are saved [path]
    @param: aug_number = how many times to augment the images [int]
    @param: rescale = rescale image? [boolean]
    @param: square = make the image square? [boolean]
    @param: dimension = what dimension to make the image [int]
    """
    
    print('==================================================')
    print('augmenting and rescaling images...')
    # augAndResize(typeToSave, imageDatasetFolder, temporaryFolder, saveToFolder, aug_num, rescale, square, dimension)
    # rescale(source_image_folder, destination, dimension):
    rescale(imageDatasetFolder, saveToFolder, dimension) 
    # =========================================================================
    
    
    # =========================================================================
    # Adjust bounding box information
    """
    @param: CSV_save_file_location = folder to save updated CSV_FILE_LOC [path]
    @param: aug_number = number of image augmentations [int]
    @param: augmentType = normal, selective, complete, etc. [String]
    @param: dimension = dimension image resized to [int]
    
    """
    print('==================================================')
    print('adjusting CSV to match augmentations...')
    # recordBbox(CSV_save_file_location, csvFileDestination, aug_num, augmentType, dimension)
    # =========================================================================
    

rescaleImages()
