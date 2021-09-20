# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:40:52 2021

@author: Eric Bianchi
conda env tf_1-10_py3-6
"""

import sys
from evaluation_gt import evaluation
from metrics_eval_build import metricsEval
from mAP_main import mAP_Results

parent_directory = './'
csv_gt_evaluation_file = './Train/Bbox_info_CSV_output_normal_300.csv'

# metricsEval(csv_gt_evaluation_file, parent_directory, type=1)

# module level variables ######################################################
train_image_directory = './Train/300x300/Images/'
destination_dir = './gt_train_bbox_images/'
NUM_CLASSES = 4
label_map_file = './label_map.pbtxt'
txt_file = './Metrics_Train_txt/'
###############################################################################

evaluation(train_image_directory, destination_dir, NUM_CLASSES, label_map_file,  txt_file, 0.10)
