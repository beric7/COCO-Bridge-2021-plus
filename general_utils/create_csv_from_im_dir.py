# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 10:32:45 2021

@author: Admin
"""
import os
import csv
from tqdm import tqdm

def build_image_file_list(TEST_IMAGES_DIR):
    
    image_file_paths = []
    image_names = []
    
    for imageFileName in os.listdir(TEST_IMAGES_DIR):
        if imageFileName.endswith(".jpg"):
            image_file_paths.append(TEST_IMAGES_DIR + imageFileName)
            image_names.append(imageFileName)
        elif imageFileName.endswith(".JPG"):
            image_file_paths.append(TEST_IMAGES_DIR + imageFileName)
            image_names.append(imageFileName)
        elif imageFileName.endswith(".png"):
            image_file_paths.append(TEST_IMAGES_DIR + imageFileName)
            image_names.append(imageFileName)
        elif imageFileName.endswith(".jpeg"):
            image_file_paths.append(TEST_IMAGES_DIR + imageFileName)
            image_names.append(imageFileName)
        
        
    return image_file_paths, image_names

def array_to_csv(array, fields, destination):
    
    with open(destination, 'w', newline='') as f:
        thewriter = csv.writer(f)
        thewriter.writerow(fields) 
        for i in tqdm(range (0,len(array))):
            print(array[i])
            thewriter.writerow([array[i]])
        print('done.')

def create_csv_from_im_dir(src, dst):
    image_file_paths, image_names = build_image_file_list(src)
    fields = ['Image Name']
    print(image_names)
    print(fields)
    array_to_csv(image_names, fields, dst)