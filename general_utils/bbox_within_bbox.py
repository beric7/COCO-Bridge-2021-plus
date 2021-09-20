# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 09:59:24 2019

@author: Eric Bianchi

TENSORFLOW: 1.10.0
PYTHON: 3.6.0
OPENCV2: newest version
Pillow: newest version


"""

import numpy as np
import os
import tensorflow as tf
print(tf.__version__)
import cv2
import sys

# When operating in ubuntu OS
#sys.path.insert(0, "/media/bianchi/Elements/Scraping_Python_Pkg/")
# When operating in windows OS
sys.path.insert(0, "D://Scraping_Python_Pkg/")
# from utils import label_map_util
from utils import visualization_utils_eric as vis_util
from eval_bbox import eval_bbox
from class_list import class_list
from model import Model

def main(cur, model):

    TEST_IMAGE_DIR = cur + "DATA/evaluation_img"

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
        
    analyzeIm(model, cur, image_names, imageFilePaths)
    
# end main
###############################################################################
def analyzeIm (model, cur, image_names, imageFilePaths):
    cur_name = 0
    for image_path in imageFilePaths:
        
        print("Evaluating image: " + image_names[cur_name])
        image_np = cv2.imread(image_path)
        
        # print(image_np.shape[1])

        if image_np is None:
            print("error reading file ")
            continue
        #end if
        
        # For each image we want to keep track of the lists of bbox for each
        # model and super-impose them at the same time.
        cat_index_list = []
        scores_list = []
        boxes_list = []
        classes_list = []
        detected_obj_list = []
        h = int(image_np.shape[0])
        w = int(image_np.shape[1])
        
        for i in range(0, len(model)):
 
            (head, tail) = image_names[cur_name].split(".")
            currentName = head
            #open_file = cur + "/Research/Scraping Images/Threshold_%/" + mod.thresholdToString() + "_Metrics_Detect_txt/" + mod.name + "_" + currentName + ".txt"
            open_file = (cur + "/DATA/Threshold_%/"  + model[i].thresholdToString() 
                         + "_Metrics_Detect_txt/" + model[i].name + "_" + currentName + ".txt")
            
            f = open(open_file,"w+")
            f.write(str(h) + "\n")
            f.write(str(w) + "\n")
            
            scores_list, boxes_list, classes_list, cat_index_list = modelPredictImg(model[i], 
                                                                                    image_np,
                                                                                    scores_list,
                                                                                    boxes_list,
                                                                                    classes_list,
                                                                                    cat_index_list)
            
            f, detect_obj = findBoxCoord(scores_list[i], boxes_list[i], classes_list[i], h, w, image_names, cur_name, model[i], f)
            
            detected_obj_list.append(detect_obj)
            f.close()
            
            #txt2xml(image_names[cur_name], TEST_IMAGE_DIR, "Img_pdf", open_file, thres_str, cur)
            
        print("All models have evaluated the image, determining intersection")
        tfVisualization(model, image_np, scores_list, boxes_list, classes_list, cat_index_list)
        # All models have been run across the images and now we can combine statements
        # and make a story out of this. 
        cv2.imshow("image_np " + image_names[cur_name], image_np)           
        cv2.waitKey()
        
        intersection(detected_obj_list)    
        cur_name = cur_name + 1
        #end for
    # end for
    
    
def intersection(detect_object_list):
    # This is the recurssive stop
    if len(detect_object_list[0]) > 0:
        # create a list of bounding box objects in the list to compare with
        # the evaluation model
        for base_bbox in detect_object_list[0]: 
            print("")
            print ("Object of interest being compared: " + base_bbox.class_type)
            compareBBOX(base_bbox, detect_object_list)
        print("")
        print("Process for this image is complete!")       
        return print("=======================================================")
    # model did not find any objects of interest in the image
    else:
        print("Model did not find any objects of interest in this image")
        return print("=======================================================")

def compareBBOX(base_bbox, detected_object_list):
    length = len(detected_object_list)
    if length == 1:
        return print("No more boxes to compare for the " + base_bbox.class_type)
    else:
        # compare against the model[0] ALWAYS.
        for i in range (1, length):
            for compare_box in detected_object_list[i]:
                # compare each bounding box foudn from the comparing model to
                # the bounding boxes found by the evaluation model
                #
                # IOU is SET here.
                compareIOU(base_bbox, compare_box, 0.35)
            # end for
        # end for

def compareIOU(base_bbox, compare_box, ovmin):
    bb = [compare_box.xmin, compare_box.ymin, compare_box.xmax, compare_box.ymax]
    bbgt = [base_bbox.xmin, base_bbox.ymin, base_bbox.xmax, base_bbox.ymax]
    bi = [max(base_bbox.xmin,compare_box.xmin), max(base_bbox.ymin ,compare_box.ymin), 
        min(base_bbox.xmax,compare_box.xmax), min(base_bbox.ymax,compare_box.ymax)]
    iw = bi[2] - bi[0] + 1
    ih = bi[3] - bi[1] + 1
    if iw > 0 and ih > 0:
        # compute overlap (IoU) = area of intersection / area of union
        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
        ov = iw * ih / ua
        if ov > ovmin:
            if (compare_box.model == "Rating"):
                print (base_bbox.class_type + " is rated as " + compare_box.class_type)
            else:
                print (compare_box.class_type + " is on a " + base_bbox.class_type)
        """else:
            #print("Not a strong enough overlap between the " + compare_box.class_type 
            #+ "and the " + base_bbox.class_type)
    else:
        #print("The " + base_bbox.class_type + " bounding box did not overlap with "+ compare_box.class_type + " at all.")
        """
                      
def modelPredictImg(mod, image_np, scores_list, boxes_list, classes_list, cat_index_list):
    
    detection_graph = mod.detection_graph
    detection_graph.as_default()
    sess = tf.Session(graph=detection_graph)

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
    
    scores_list.append([scores])
    boxes_list.append([boxes])
    classes_list.append([classes])
    
    category_index = mod.getDict()
    # track indexes
    cat_index_list.append(category_index)

    return scores_list, boxes_list, classes_list, cat_index_list 


def findBoxCoord(scores, boxes, classes, h, w, image_names, cur_name, mod, f):
    
    i = 0
    detect_obj = []
    while (i < 100 and (scores[0][0][i]) > (mod.threshold)):             
        ymin = int((boxes[0][0][i][0])*h)
        xmin = int((boxes[0][0][i][1])*w)
        ymax = int((boxes[0][0][i][2])*h)
        xmax = int((boxes[0][0][i][3])*w)
    
        #print(classes[0][i])
        class_cur = class_list(classes[0][0][i], mod.type).ID
        #print(class_cur.ID)
    
        score = (scores[0][0][i])
        
        cur_bbox = eval_bbox(image_names[cur_name], class_cur, mod.type, score , xmin, ymin, xmax, ymax)
        detect_obj.append(cur_bbox)
        f.write(detect_obj[i].toString() + "\n")
        print(detect_obj[i].toString())
      
        i = i + 1   
        
    return f, detect_obj               

def tfVisualization(model, image_np, scores_list, boxes_list, classes_list, cat_index_list):
    for i in range(0, len(model)):
        vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                           np.squeeze(boxes_list[i]),
                                                           np.squeeze(classes_list[i]).astype(np.int32),
                                                           np.squeeze(scores_list[i]),
                                                           cat_index_list[i],
                                                           model[i],
                                                           min_score_thresh=model[i].threshold,
                                                           use_normalized_coordinates=True)
###############################################################################
###############################################################################


if __name__ == "__main__":
    
    # Here we initialize our model that we want to run on top of each other. 
    model = []
    DRIVE = "D://"
    pwd = DRIVE + "MODELS/BridgeDetailModels/"
    #pwd="E://BridgeDetailModels/"
    # THIS IS AN EXAMPLE: I am using this 
    #--------------------------------------------------------------------------
    model.append(Model("#6-NFS_4c_5000s1e", 0.25, "Structural Details", pwd, 'Orange'))
    model.append(Model("#6-NFS_4c_5000s1e", 0.25, "Structural Cracks", pwd, 'Fuchsia'))
    model.append(Model("#6-NFS_4c_5000s1e", 0.25, "Rating", pwd, 'DarkSeaGreen'))
    #--------------------------------------------------------------------------
    
    main(DRIVE, model)
