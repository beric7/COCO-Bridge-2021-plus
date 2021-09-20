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


# module level variables ######################################################
test_image_directory = './Evaluation/300x300/Images/'
destination_dir = './gt_bbox_images/'
NUM_CLASSES = 4
label_map_file = './label_map.pbtxt'
###############################################################################

evaluation(validation_image_directory, destination_dir, NUM_CLASSES, label_map_file,  0.10)
