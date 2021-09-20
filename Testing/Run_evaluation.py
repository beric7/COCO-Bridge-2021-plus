# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 15:15:34 2018

@author: Eric Bianchi
conda env tf_1-10_py3-6
"""

import sys

from evaluation import evaluation
# main(parent_directory, validation_image_directory, NUM_CLASSES, label_map_file, threshold, thres_str)
from metrics_eval_build import metricsEval
# (csv_gt_evaluation_file)
from mAP_main import mAP_Results
# mAP_Results(parent_directory, validation_image_directory, minOverLap, thres_str)

# module level variables ######################################################
parent_directory = './'
destination_dir = './predicted_images/'
test_image_directory = './Evaluation/300x300/Images/'
NUM_CLASSES = 4
label_map_file = './label_map.pbtxt'
# threshold = 0.35
# thres_str = '35%'

csv_gt_evaluation_file = './Evaluation/300x300/bbox/csv/' + 'Bbox_info_CSV_output_normal_300.csv'
###############################################################################

metricsEval(csv_gt_evaluation_file, parent_directory)

'''
evaluation(parent_directory, test_image_directory, destination_dir, NUM_CLASSES, label_map_file,  0.01, '1%')
evaluation(parent_directory, test_image_directory, destination_dir, NUM_CLASSES, label_map_file,  0.05, '5%')
evaluation(parent_directory, test_image_directory, destination_dir, NUM_CLASSES, label_map_file,  0.10, '10%')
evaluation(parent_directory, test_image_directory, destination_dir, NUM_CLASSES, label_map_file,  0.25, '25%')
'''

mAP_Results(parent_directory, test_image_directory, 0.35, "1%")
mAP_Results(parent_directory, test_image_directory, 0.35, "5%")
mAP_Results(parent_directory, test_image_directory, 0.35, "10%")
mAP_Results(parent_directory, test_image_directory, 0.35, "25%")

mAP_Results(parent_directory, test_image_directory, 0.50, "1%")
mAP_Results(parent_directory, test_image_directory, 0.50, "5%")
mAP_Results(parent_directory, test_image_directory, 0.50, "10%")
mAP_Results(parent_directory, test_image_directory, 0.50, "25%")

mAP_Results(parent_directory, test_image_directory, 0.75, "1%") 
mAP_Results(parent_directory, test_image_directory, 0.75, "5%")
mAP_Results(parent_directory, test_image_directory, 0.75, "10%")
mAP_Results(parent_directory, test_image_directory, 0.75, "25%")

