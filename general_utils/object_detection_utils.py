# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 18:32:21 2020

@author: Eric Bianchi
"""
import tensorflow as tf
import sys
from TXT_to_XML import txt2xml
from utils import label_map_util
from utils import visualization_utils_eric as vis_util
from eval_bbox import eval_bbox
from class_list import class_list
import numpy as np
import os
import cv2

def loadFrozenGraph(FROZEN_INFERENCE_GRAPH_LOC):
    # load a (frozen) TensorFlow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(FROZEN_INFERENCE_GRAPH_LOC, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        # end with
    # end with
    return detection_graph

def loadLabels(LABELS_LOC, NUM_CLASSES):
    # Loading label map
    # Label maps map indices to category names, so that when our convolution network 
    # predicts `5`, we know that this corresponds to `airplane`.  
    # Here we use internal utility functions, but anything that 
    # returns a dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(LABELS_LOC)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    return categories, category_index

def makeDetectionPred(detection_graph, image_np, sess, category_index, threshold):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    return boxes, scores, classes

def visualize(image_np, model, boxes, scores, classes, category_index, threshold):
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                       np.squeeze(boxes),
                                                       np.squeeze(classes).astype(np.int32),
                                                       np.squeeze(scores),
                                                       category_index,
                                                       model,
                                                       min_score_thresh=threshold,
                                                       use_normalized_coordinates=True)
    return image_np
    

def buildImageFileList(BASE, TEST_IMAGES_DIR, sort_ID_string):
    
    imageFilePaths = []
    image_names = []
    
    UNUSABLE_FILETYPE_DIR = "DATA/Extraction/Sorted Data/"
    
    for imageFileName in os.listdir(TEST_IMAGES_DIR):
        if imageFileName.endswith(".jpg"):
            imageFilePaths.append(TEST_IMAGES_DIR + imageFileName)
            image_names.append(imageFileName)
        if imageFileName.endswith(".JPG"):
            imageFilePaths.append(TEST_IMAGES_DIR + imageFileName)
            image_names.append(imageFileName)
        if imageFileName.endswith(".png"):
            imageFilePaths.append(TEST_IMAGES_DIR + imageFileName)
            image_names.append(imageFileName)
        if imageFileName.endswith(".jpeg"):
            imageFilePaths.append(TEST_IMAGES_DIR + imageFileName)
            image_names.append(imageFileName)
            
        # TODO: This does remove the file of interest... may not want to do this in the future. 
        if imageFileName.endswith(".emf"):
            os.rename(BASE + TEST_IMAGES_DIR + imageFileName, BASE + UNUSABLE_FILETYPE_DIR + sort_ID_string + "/EMF/" + imageFileName)
        if imageFileName.endswith(".wmf"):
            os.rename(BASE + TEST_IMAGES_DIR + imageFileName, BASE + UNUSABLE_FILETYPE_DIR + sort_ID_string + "/WMF/" + imageFileName)  
        if imageFileName.endswith(".gif"):
            os.rename(BASE + TEST_IMAGES_DIR + imageFileName, BASE + UNUSABLE_FILETYPE_DIR + sort_ID_string + "/GIF/" + imageFileName)
        if imageFileName.endswith(".tif"):
            os.rename(BASE + TEST_IMAGES_DIR + imageFileName, BASE + UNUSABLE_FILETYPE_DIR + sort_ID_string + "/TIF/" + imageFileName)
        if imageFileName.endswith(".tiff"):
            os.rename(BASE + TEST_IMAGES_DIR + imageFileName, BASE + UNUSABLE_FILETYPE_DIR + sort_ID_string + "/TIFF/" + imageFileName) 
        if imageFileName.endswith(".wdp"):
            os.rename(BASE + TEST_IMAGES_DIR + imageFileName, BASE + UNUSABLE_FILETYPE_DIR + sort_ID_string + "/WDP/" + imageFileName)

    return imageFilePaths, image_names

def writePredToXML(image_names, thres_str, image_np, threshold, scores, classes, boxes, TEST_IMAGE_DIR, cur):
    (head, tail) = image_names[cur_name].split(".")
    currentName = head
    open_file = cur + "/Threshold_%/" + thres_str + "_Metrics_Detect_txt/" + currentName + ".txt"
    
    f = open(open_file,"w+")
    
    h = int(image_np.shape[0])
    w = int(image_np.shape[1])
    f.write(str(h) + "\n")
    f.write(str(w) + "\n")
    
    i = 0
    detect_obj = []
    while (i < 100 and (scores[0][i]) > (threshold)):
        
        ymin = int((boxes[0][i][0])*h)
        xmin = int((boxes[0][i][1])*w)
        ymax = int((boxes[0][i][2])*h)
        xmax = int((boxes[0][i][3])*w)
    
        #print(classes[0][i])
        class_cur = class_list(classes[0][i]).ID
        #print(class_cur.ID)
    
        score = (scores[0][i])
        
        cur_bbox = eval_bbox(image_names[cur_name], class_cur, score , xmin, ymin, xmax, ymax)
        detect_obj.append(cur_bbox)
        f.write(detect_obj[i].toString() + "\n")
  
        i = i + 1
   
    f.close()
    
    txt2xml(image_names[cur_name], TEST_IMAGE_DIR, "Img_pdf", open_file, thres_str, cur)