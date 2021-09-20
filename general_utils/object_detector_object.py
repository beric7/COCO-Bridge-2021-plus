# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 20:07:40 2020

@author: Eric Bianchi
"""

class objectDetector(object):
    
    def __init__(self,Model, BASE, TEST_IMAGE_DIR):
        self.Model = Model
        self.model_dir = BASE + "MODELS/BridgeDetailModels/" + Model + "/Pre-Processing"
        self.FROZEN_INFERENCE_GRAPH_LOC = self.model_dir + "/inference_graph/frozen_inference_graph.pb"
        self.LABELS_LOC = self.model_dir + "/" + "label_map.pbtxt"
        self.TEST_IMAGE_DIR = TEST_IMAGE_DIR
        self.BASE = BASE
        self.cur = BASE + "DATA/"