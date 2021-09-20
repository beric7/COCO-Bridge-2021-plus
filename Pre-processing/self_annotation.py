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

Based on code from Chris Dahms
"""

import numpy as np
import os
import tensorflow as tf
import cv2
import sys

sys.path.insert(0, "E://Scraping_Python_Pkg")
from utils import label_map_util
from utils import visualization_utils as vis_util
from eval_bbox import eval_bbox
from class_list import class_list
from TXT_to_XML import txt2xml

def main(NUM_CLASSES, Model, threshold, thres_str):
    ###########################################################################
    pwd = "E://BridgeDetailModels/" + Model + "/Pre-Processing"
    cur = "E://DATA"
    TEST_IMAGE_DIR = cur + "/Test"
    FROZEN_INFERENCE_GRAPH_LOC = pwd + "/inference_graph/frozen_inference_graph.pb"
    LABELS_LOC = pwd + "/" + "label_map.pbtxt"
    imageFileName = ""
    ###########################################################################
    print("starting program . . .")
    print("Tensorflow Version: " + str(tf.__version__))
    if not checkIfNecessaryPathsAndFilesExist(TEST_IMAGE_DIR, FROZEN_INFERENCE_GRAPH_LOC, 
                                              LABELS_LOC):
        return
    # end if

    # this next comment line is necessary to avoid a false PyCharm warning
    # noinspection PyUnresolvedReferences
    if tf.__version__ < '1.5.0':
        raise ImportError('Please upgrade your tensorflow installation to v1.5.* or later!')
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
        if imageFileName.endswith(".jpg"):
            imageFilePaths.append(TEST_IMAGE_DIR + "/" + imageFileName)
        if imageFileName.endswith(".JPG"):
            imageFilePaths.append(TEST_IMAGE_DIR + "/" + imageFileName)
        if imageFileName.endswith(".png"):
            imageFilePaths.append(TEST_IMAGE_DIR + "/" + imageFileName)
        if imageFileName.endswith(".jpeg"):
            imageFilePaths.append(TEST_IMAGE_DIR + "/" + imageFileName)
        # end if
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
                    class_cur = class_list(classes[0][i], "Structural Details").ID
                    #print(class_cur.ID)
                
                    score = (scores[0][i])
                    
                    cur_bbox = eval_bbox(image_names[cur_name], class_cur, score , xmin, ymin, xmax, ymax)
                    detect_obj.append(cur_bbox)
                    f.write(detect_obj[i].toString() + "\n")
          
                    i = i + 1
       
                f.close()
                
                txt2xml(image_names[cur_name], TEST_IMAGE_DIR, "Img_pdf", open_file, thres_str, cur)
                
                #cv2.imshow("image_np " + image_names[cur_name], image_np)           
                #cv2.waitKey()
                
                cur_name = cur_name + 1
            # end for
        # end with
    # end with
# end main
#######################################################################################################################
def writeResultOnImage(openCVImage, resultText):
    # ToDo: this function may take some further fine-tuning to show the text well given any possible image size

    imageHeight, imageWidth, sceneNumChannels = openCVImage.shape

    # choose a font
    fontFace = cv2.FONT_HERSHEY_TRIPLEX

    # chose the font size and thickness as a fraction of the image size
    if imageWidth < 800:
        fontScale = imageWidth/800
    else:
        fontScale = 1
        
    fontThickness = .5

    # make sure font thickness is an integer, if not, the OpenCV functions that use this may crash
    fontThickness = int(fontThickness)

    upperLeftTextOriginX = int(imageWidth * 0.05)
    upperLeftTextOriginY = int(imageHeight * 0.05)

    textSize, baseline = cv2.getTextSize(resultText, fontFace, fontScale, fontThickness)
    textSizeWidth, textSizeHeight = textSize

    # calculate the lower left origin of the text area based on the text area center, width, and height
    lowerLeftTextOriginX = upperLeftTextOriginX
    lowerLeftTextOriginY = upperLeftTextOriginY + textSizeHeight
    
    rectangle_bgr = (255,255,0)
    box_coords = ((upperLeftTextOriginX - 5, upperLeftTextOriginY - 5), (upperLeftTextOriginX + textSizeWidth + 5, upperLeftTextOriginY + textSizeHeight + 5))
    cv2.rectangle(openCVImage, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)

    # write the text on the image
    cv2.putText(openCVImage, resultText, (lowerLeftTextOriginX, lowerLeftTextOriginY), fontFace, fontScale, SCALAR_BLUE, fontThickness)
# end function
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