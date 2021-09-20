# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 20:07:40 2020

@author: Eric Bianchi
"""
import tensorflow as tf
import shutil

class binaryClassifier(object):
    
    def __init__(self, TEST_IMAGE_DIR, MODEL_DIR):
        self.MODEL_DIR = MODEL_DIR
        self.TEST_IMAGE_DIR = TEST_IMAGE_DIR
    
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # HELPER METHOD: Try except statements
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
    def checkDirTryCatch(path):
        try:
            shutil.rmtree(path)
        except:
            print(path + " already removed")
    
    def prepareGraphs(self):
        print("preparing graphs . . .")
        
        # Set model directory paths
        MODEL_DIR = self.MODEL_DIR
        
        # Obtain trained model files
        RETRAINED_LABELS_TXT_FILE_LOC = MODEL_DIR + "/Classification/" + "retrained_labels.txt"
        RETRAINED_GRAPH_PB_FILE_LOC = MODEL_DIR + "/Classification/" + "retrained_graph.pb"
    
        # get a list of classifications from the labels file
        classifications = []
        # for each line in the label file . . .
        for currentLine in tf.gfile.GFile(RETRAINED_LABELS_TXT_FILE_LOC):
            # remove the carriage return
            classification = currentLine.rstrip()
            # and append to the list
            classifications.append(classification)
        # end for
    
        # Show the classifications to prove out that we were able to read the label file successfully
        print("classifications = " + str(classifications))
    
    
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(RETRAINED_GRAPH_PB_FILE_LOC, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                
        # end with
        return classifications, detection_graph