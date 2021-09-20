# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:19:05 2019

@author: Eric Bianchi
"""
import tensorflow as tf

class Model(object):

    def __init__(self, Model, threshold, graph_name, pwd, color):
        pwd = pwd + Model + "/Pre-Processing"
        self.graph = pwd + "/inference_graph/frozen_inference_graph.pb"
        self.threshold = threshold
        self.type = graph_name
        self.name = Model
        self.detection_graph = self.loadGraph()
        self.color = color
        
    def thresholdToString(self):
        return str(int(self.threshold*100)) + "%"

    def loadGraph(self):
        # load a (frozen) TensorFlow model into memory
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.graph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")
        return detection_graph
            # end with
        # end with):
        
        
    def getDict(self):
        if self.type == "Structural Details":
            dictionary = {1: "Gusset Plate Connection",
                          2: "Out of Plane Stiffener",
                          3: "Cover Plate Termination",
                          4: "Bearing"}
        elif self.type == "Structural Cracks":
            dictionary = {1: "Crack Damage",
                          2: "Crack Damage",
                          3: "Crack Damage",
                          4: "Crack Damage"}
        elif self.type == "Rating":
            dictionary = {1: "Severe (4)",
                          2: "Poor (3)",
                          3: "Fair (2)",
                          4: "Good (1)"}   
        return dictionary
