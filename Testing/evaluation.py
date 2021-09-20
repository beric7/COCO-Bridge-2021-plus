# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 09:59:24 2019

@author: Eric Bianchi
reviewed: 6/17/2020
"""

# test.py
"""
Created on Sat Nov  3 15:15:34 2018

@author: Eric Bianchi
"""

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

def evaluation(parent_directory, test_image_directory, NUM_CLASSES, label_map_file, threshold, thres_str):
    ###########################################################################
    TEST_IMAGE_DIR = test_image_directory
    FROZEN_INFERENCE_GRAPH_LOC = parent_directory + "/frozen_inference_graph.pb"
    LABELS_LOC = label_map_file
    imageFileName = ""
    ###########################################################################
    print("starting program . . .")
    if not checkIfNecessaryPathsAndFilesExist(TEST_IMAGE_DIR, FROZEN_INFERENCE_GRAPH_LOC, 
                                              LABELS_LOC):
        return
    # end if

    # this next comment line is necessary to avoid a false PyCharm warning
    # noinspection PyUnresolvedReferences
    # if tf.__version__ < '1.5.0':
    #     raise ImportError('Please upgrade your tensorflow installation to v1.5.* or later!')
    # end if

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
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            for image_path in imageFilePaths:

                # print(image_path)
                
                print(image_names[cur_name])
                image_np = cv2.imread(image_path)
                
                # print(image_np.shape[1])

                if image_np is None:
                    print("error reading file ")
                    continue
                #end if
                h = int(image_np.shape[0])
                w = int(image_np.shape[1])

                if (w > 4000 or h > 4000):
                    w = int(w/10)
                    h = int(h/10)
                    image_np = cv2.resize(image_np,(h,w))
                else: 
                    image_np = image_np

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
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                                    np.squeeze(boxes),
                                                                    np.squeeze(classes).astype(np.int32),
                                                                    np.squeeze(scores),
                                                                    category_index,
                                                                    use_normalized_coordinates=True,
                                                                    line_thickness=1, min_score_thresh= threshold)

                if not os.path.exists(parent_directory + "/Threshold_%/" + thres_str + "_Metrics_Detect_txt/"): # if it doesn't exist already
                    os.makedirs(parent_directory + "/Threshold_%/" + thres_str + "_Metrics_Detect_txt/")
                f = open(parent_directory + "/Threshold_%/" + thres_str + "_Metrics_Detect_txt/" + image_names[cur_name] + ".txt","w+")
                
                i = 0
                detect_obj = []
                while (i < 100 and (scores[0][i]) > (threshold)):
                    
                    ymin = int((boxes[0][i][0])*h)
                    xmin = int((boxes[0][i][1])*w)
                    ymax = int((boxes[0][i][2])*h)
                    xmax = int((boxes[0][i][3])*w)
                
                    #print(classes[0][i])
                    # class_list() takes two positional arguments at the moment. Open to see in general_utils folder 
                    class_cur = class_list(classes[0][i], "Structural Details").ID
                    #print(class_cur.ID)
                
                    score = (scores[0][i])
                    
                    # (ID, class_type, model, score, xmin, ymin, xmax, ymax)
                    cur_bbox = eval_bbox(image_names[cur_name], class_cur, score, xmin, ymin, xmax, ymax)
                    detect_obj.append(cur_bbox)
                    f.write(detect_obj[i].toString() + "\n")
          
                    i = i + 1
       
                f.close()
                # cv2.imshow("image_np " + image_names[cur_name], image_np)           
                # cv2.waitKey()
                
                cur_name = cur_name + 1
            # end for
        # end with
    # end with
# end main

#######################################################################################################################
def checkIfNecessaryPathsAndFilesExist(TEST_IMAGE_DIR, FROZEN_INFERENCE_GRAPH_LOC,LABELS_LOC):
    if not os.path.exists(TEST_IMAGE_DIR):
        print('ERROR: TEST_IMAGE_DIR "' + TEST_IMAGE_DIR + '" does not seem to exist')
        return False
    # end if

    # ToDo: check here that the test image directory contains at least one image

    if not os.path.exists(FROZEN_INFERENCE_GRAPH_LOC):
        print('ERROR: FROZEN_INFERENCE_GRAPH_LOC "' + FROZEN_INFERENCE_GRAPH_LOC + '" does not seem to exist')
        print('was the inference graph exported successfully?')
        return False
    # end if

    if not os.path.exists(LABELS_LOC):
        print('ERROR: the label map file "' + LABELS_LOC + '" does not seem to exist')
        return False
    # end if

    return True
# end function

#######################################################################################################################
