# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:56:09 2021

@author: Eric Bianchi
"""
# -*- coding: utf-8 -*-


import numpy as np
import os
import tensorflow as tf
import cv2
import sys

sys.path.insert(0, "../")
from utils import label_map_util
from utils import visualization_utils as vis_util
sys.path.insert(0, "../general_utils/")
from eval_bbox import eval_bbox
from class_list import class_list
from txt_to_xml import txt2xml

def evaluation(test_image_directory, destination_dir, NUM_CLASSES, label_map_file, txt_file, threshold):
    ###########################################################################
    TEST_IMAGE_DIR = test_image_directory
    LABELS_LOC = label_map_file
    imageFileName = ""
    ###########################################################################
    print("starting program . . .")
    
    if not os.path.exists(destination_dir): # if it doesn't exist already
        os.makedirs(destination_dir)
    # Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(LABELS_LOC)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    imageFilePaths = []
    image_names = []
    for imageFileName in os.listdir(TEST_IMAGE_DIR):
        imageFilePaths.append(TEST_IMAGE_DIR + "/" + imageFileName)
        image_names.append(imageFileName)
    # end for
    cur_name = 0
    for image_path in imageFilePaths:

        print(image_names[cur_name])
        image_np = cv2.imread(image_path)

        if image_np is None:
            print("error reading file ")
            continue
        #end if
        
        image_np = cv2.resize(image_np,(300,300))
        
        boxes = []
        classes = []
        scores = None
        
        metrics_gt = txt_file
        with open(metrics_gt+image_names[cur_name]+".txt", "r") as filestream:
            for line in filestream:
                currentline = line.split(",")
                boxes.append([int(currentline[2]), int(currentline[1]), int(currentline[4]), int(currentline[3])])
                for item in category_index:
                    # print(category_index[item])
                    if category_index[item]['name'] == currentline[0]:
                        class_name = int(category_index[item]['id'])
                #class_name = next(item for item in category_index if item["name"] == currentline[0])
                classes.append(class_name)
        boxes = np.array(boxes)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array_from_text(image_np,
                                                            boxes,
                                                            classes,
                                                            scores,
                                                            category_index,
                                                            agnostic_mode = True,
                                                            use_normalized_coordinates=True,
                                                            line_thickness=1, min_score_thresh= threshold)

        cv2.imwrite(destination_dir + image_names[cur_name], image_np)
        
        cur_name = cur_name + 1

# end main
