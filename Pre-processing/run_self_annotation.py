# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 09:58:36 2019

@author: Eric Bianchi
reviewed: 6/17/2020
"""

import sys
sys.path.insert(0, "E://Scraping_Python_Pkg")
from test_self_annotation import main


# ============================================================================
Model = "#6-NFS_4c_5000s1e"
# ============================================================================
# module level variables ######################################################
NUM_CLASSES = 4
###############################################################################

main(NUM_CLASSES, Model, 0.10, "10%")