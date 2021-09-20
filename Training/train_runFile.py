# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:12:24 2020

@author: Eric Bianchi
"""


# SET PATH TO MODLES FOR WHERE CODE RUNS
PATH_TO_MODEL = '/home/beric7/COCO-Bridge-2020/MODELS/BridgeDetailModels/original_after_editing/'
VARIATION = 'var1'
SAVE_TRAINING_DATA_HERE = 'training_param_variations/' + VARIATION + '/'

import sys
sys.path.insert(0, PATH_TO_MODEL)
from ssd_train import train

"""
@param: SAVE_TRAINING_DATA_HERE = path to where variation will be saved 
@param: PATH_TO_MODEL = path to specific dataset
"""
train(PATH_TO_MODEL, SAVE_TRAINING_DATA_HERE)
