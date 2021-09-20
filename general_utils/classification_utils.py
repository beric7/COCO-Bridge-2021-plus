# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 16:40:47 2020

@author: Eric Bianchi
"""
import shutil
import os
import numpy as np
import tensorflow as tf
import cv2

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Try except statements
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
def checkDirTryCatch(path):
    try:
        shutil.rmtree(path)
    except:
        print(path + " already removed")
        
        
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Remove folder contents
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def removeCon(base_path):
    
    # Remove the existing folders   
    checkDirTryCatch(base_path + '/EMF')
    checkDirTryCatch(base_path + '/GIF')
    checkDirTryCatch(base_path + '/TIFF') 
    checkDirTryCatch(base_path + '/WDP')
    checkDirTryCatch(base_path + '/WMF')
    checkDirTryCatch(base_path + '/UnableToOpen')
    
    os.mkdir(base_path + '/EMF')
    os.mkdir(base_path + '/GIF')
    os.mkdir(base_path + '/TIFF') 
    os.mkdir(base_path + '/WDP')
    os.mkdir(base_path + '/WMF')
    os.mkdir(base_path + '/UnableToOpen')

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
    cv2.putText(openCVImage, resultText, (lowerLeftTextOriginX, lowerLeftTextOriginY), fontFace, fontScale, (255, 0, 0) , fontThickness)
# end function

def overwrite(original, output):
    try:
        os.rename(original, output)
    except WindowsError:
        os.remove(output)
        os.rename(original, output)
        
def confidenceSort(onMostLikelyPrediction,confidence,strClassification,cur_name,image_path,DRIVE,image_names,countList, sort_ID, sort_ID_string ,upperBound,lowerBound,x,y):
    if onMostLikelyPrediction and (upperBound > confidence > lowerBound):
        string_upperBound = str(int(100*upperBound))
        string_lowerBound = str(int(100*lowerBound))
        scoreAsAPercent = confidence * 100.0
        # print("the object appears to be a " + strClassification + ", " +
        #       "{0:.2f}".format(scoreAsAPercent) + "% confidence  (" + str(cur_name) + ")")
        
        if strClassification == (sort_ID.lower()): 
            shutil.copyfile(image_path, DRIVE + "DATA/Extraction/Sorted Data/" +
                            sort_ID_string + "/#" + sort_ID +"/" + string_lowerBound +
                            "%-" + string_upperBound + "%/" + image_names[cur_name])   
            countList[x][y] = countList[x][y] + 1
            
        else:
            shutil.copyfile(image_path, DRIVE + "DATA/Extraction/Sorted Data/" +
                            sort_ID_string + "/#Non" + sort_ID +"/" + string_lowerBound +
                            "%-" + string_upperBound + "%/" + image_names[cur_name])      
            countList[x+1][y] = countList[x+1][y] + 1
            
        onMostLikelyPrediction = False
        cur_name = cur_name + 1
    return cur_name, onMostLikelyPrediction

def confidenceSort_(image_path,BASE,image_name,confidence, sort_ID, sort_ID_string, upperBound,lowerBound):
    string_upperBound = str(int(100*upperBound))
    string_lowerBound = str(int(100*lowerBound))
    dest = BASE + "DATA/Extraction/Sorted Data/" + sort_ID_string + "/#" + sort_ID + "/" + string_lowerBound + "%-" + string_upperBound + "%/" + image_name
    
    from os import path
    
    if path.exists(dest):
        print(image_name + ': already exists')
    else:
        shutil.copyfile(image_path, dest)
        print('sorted data: ' + image_name + 'to #' + sort_ID + ' ' + string_upperBound)
        

def buildImageFileList(TEST_IMAGES_DIR):
    
    imageFilePaths = []
    image_names = []
    
    for imageFileName in os.listdir(TEST_IMAGES_DIR):
        if imageFileName.endswith(".jpg"):
            imageFilePaths.append(TEST_IMAGES_DIR + imageFileName)
            image_names.append(imageFileName)
        elif imageFileName.endswith(".JPG"):
            imageFilePaths.append(TEST_IMAGES_DIR + imageFileName)
            image_names.append(imageFileName)
        elif imageFileName.endswith(".png"):
            imageFilePaths.append(TEST_IMAGES_DIR + imageFileName)
            image_names.append(imageFileName)
        elif imageFileName.endswith(".jpeg"):
            imageFilePaths.append(TEST_IMAGES_DIR + imageFileName)
            image_names.append(imageFileName)
        
        
    return imageFilePaths, image_names



def makePrediction(sess, openCVImage, classifications):
    # get the final tensor from the graph
    finalTensor = sess.graph.get_tensor_by_name('final_result:0')
    activations = sess.graph.get_tensor_by_name('mixed_10/join:0')

    # convert the OpenCV image (numpy array) to a TensorFlow image
    tfImage = np.array(openCVImage)[:, :, 0:3]
    
    # run the network to get the predictions
    predictions = sess.run(finalTensor, {'DecodeJpeg:0': tfImage})
    
    activations = sess.run(activations, {'DecodeJpeg:0': tfImage})
    # sort predictions from most confidence to least confidence
    sortedPredictions = predictions[0].argsort()[-len(predictions[0]):][::-1]

    prediction = sortedPredictions[0]
    #*****print("---------------------------------------")
    # keep track of if we're going through the next for loop for the first time so we can show more info about
    # the first prediction, which is the most likely prediction (they were sorted descending above)
    onMostLikelyPrediction = True
    
    strClassification = classifications[prediction]

    # get confidence, then get confidence rounded to 2 places after the decimal
    confidence = predictions[0][prediction]
    
    return prediction, onMostLikelyPrediction, sortedPredictions, strClassification, confidence

def addResultToImList(image_path, openCVImage, strClassification, confidence, cur_name):
    # get the score as a %
    scoreAsAPercent = confidence * 100.0
    # show the result to std out
    writeResultOnImage(openCVImage, strClassification + ", " + "{0:.2f}".format(scoreAsAPercent) + "% confidence.")
    
    return openCVImage


def showResultOnImage(image_path, openCVImage, strClassification, confidence, cur_name):
    # get the score as a %
    scoreAsAPercent = confidence * 100.0
    # show the result to std out
    print("the object appears to be a " + strClassification + ", " + "{0:.2f}".format(scoreAsAPercent) + "% confidence.  (" + str(cur_name) + ")")
    writeResultOnImage(openCVImage, strClassification + ", " + "{0:.2f}".format(scoreAsAPercent) + "% confidence.")
    
    # finally we can show the OpenCV image
    cv2.imshow(image_path, openCVImage)
    # mark that we've show the most likely prediction at this point so the additional information in
    # this if statement does not show again for this image             
    cv2.waitKey()
    # after a key is pressed, close the current window to prep for the next time around
    cv2.destroyAllWindows()
    
def makeCountList():
    non0 = 0
    non25 = 0
    non65 = 0
    non75 = 0
    non90 = 0
    non95 = 0
    
    yes0 = 0
    yes25 = 0
    yes65 = 0
    yes75 = 0
    yes90 = 0
    yes95 = 0
    
    countList = [[0, 0, 0, 0, 0, 0, 0],
                       [0,yes0,yes25, yes65, yes75, yes90, yes95],
                       [0,non0,non25, non65, non75, non90, non95]]
    return countList


def sortPrediction_(image_path, BASE, confidence,strClassification,image_name, sort_ID, sort_ID_string):
    
    if confidence < 0.2:     
        confidenceSort_(image_path,BASE,image_name,confidence,sort_ID,sort_ID_string, 0.2,0) 
    elif 0.2 < confidence < 0.4:
        confidenceSort_(image_path,BASE,image_name,confidence,sort_ID,sort_ID_string ,0.4,0.2)    
    elif 0.4 < confidence < 0.6:
        confidenceSort_(image_path,BASE,image_name,confidence,sort_ID,sort_ID_string ,0.6,0.4)
    elif 0.6 < confidence < 0.8:
        confidenceSort_(image_path,BASE,image_name,confidence,sort_ID,sort_ID_string ,0.8,0.6)
    elif 0.8 < confidence < 1.0:
        confidenceSort_(image_path,BASE,image_name,confidence,sort_ID,sort_ID_string ,1.0,0.8)

def sortPrediction(onMostLikelyPrediction, confidence,strClassification,image_path,
        BASE, image_names, countList, sort_ID, sort_ID_string, cur_name):
            
    cur_name, onMostLikelyPrediction = confidenceSort(
        onMostLikelyPrediction, confidence,
        strClassification, cur_name, image_path,
        BASE, image_names, countList, sort_ID, sort_ID_string ,0.2,0.0,1,2)
    
    cur_name, onMostLikelyPrediction = confidenceSort(
        onMostLikelyPrediction, confidence,
        strClassification, cur_name, image_path,
        BASE, image_names, countList, sort_ID, sort_ID_string ,0.4,0.2,1,3)
    
    cur_name, onMostLikelyPrediction = confidenceSort(
        onMostLikelyPrediction, confidence,
        strClassification, cur_name, image_path,
        BASE, image_names, countList, sort_ID, sort_ID_string ,0.6,0.4,1,4)
    
    cur_name, onMostLikelyPrediction = confidenceSort(
        onMostLikelyPrediction, confidence,
        strClassification, cur_name, image_path,
        BASE, image_names, countList, sort_ID, sort_ID_string ,0.8,0.6,1,5)
    
    cur_name, onMostLikelyPrediction = confidenceSort(
        onMostLikelyPrediction, confidence,
        strClassification, cur_name, image_path,
        BASE, image_names, countList, sort_ID, sort_ID_string ,1.0,0.8,1,6)
    

def saveAndPrintHistogram(sort_ID, countList):
    text = ( "\n"
    "\n"
    "================== " + sort_ID + " =============" + "\n"
    + sort_ID + " <25%    : " + str(countList[1][1]) + "\n"
    + sort_ID + " 25%-50% : " + str(countList[1][2]) + "\n"
    + sort_ID + " 65%-75% : " + str(countList[1][3]) + "\n"
    + sort_ID + " 75%-90% : " + str(countList[1][4]) + "\n"
    + sort_ID + " 90%-95% : " + str(countList[1][5]) + "\n"
    + sort_ID + " >95%    : " + str(countList[1][6]) + "\n"
    "================== Non-" + sort_ID + " ==================" + "\n"
    + "non" + sort_ID + " <25%        : " + str(countList[2][1]) + "\n"
    + "non" + sort_ID + " 25%-50%     : " + str(countList[2][2]) + "\n"
    + "non" + sort_ID + " 65%-75%     : " + str(countList[2][3]) + "\n"
    + "non" + sort_ID + " 75%-90%     : " + str(countList[2][4]) + "\n"
    + "non" + sort_ID + " 90%-95%     : " + str(countList[2][5]) + "\n"
    + "non" + sort_ID + " >95%        : " + str(countList[2][6]) + "\n")
    
    print(text)
    histogram = open(sort_ID + ".txt","w") 
    lines = [text]
    histogram.writelines(lines) 
    histogram.close()
    
def prettyLabel(classDirectory):
    for class_ in classDirectory:
        
        for im in directoryPath:
            shutil.copyfile(TEST_IMAGE_DIR + im, FN_PATH + im)
        print('done')

def confusionMatrix(pred, gt):
    
    if gt == 0 and pred == 0:
        result = 'TP'
    if gt == 1 and pred == 1:
        result = 'TN'
    if gt == 0 and pred == 1:
        result = 'FN'
    if gt == 1 and pred == 0:
        result = 'FP'
    return result

def sortRandomImages(src, dst, number):
    
    imageFilePaths, image_names = buildImageFileList(src, '')
    
    import random
    for i in range(0,number):
        randNumber = random.randrange(0, len(imageFilePaths), 1)
        print (randNumber)
        src_rand = imageFilePaths[randNumber]
        shutil.copy(src_rand, dst)
        imageFilePaths.remove(src_rand)

def sortImages(src, dst, image_names):

    if not os.path.exists(dst): # if it doesn't exist already
        os.makedirs(dst)
        
    for i in range(0,len(image_names)):
        print(image_names[i][0])
        image = image_names[i][0]
        
        
        shutil.copy(src+image, dst)
        
def sortXML(src, dst, image_names):

    if not os.path.exists(dst): # if it doesn't exist already
        os.makedirs(dst)
        
    for i in range(0,len(image_names)):
        print(image_names[i][0])
        xml = image_names[i][0].split('.')[0]+'.xml'

        shutil.copy(src+xml, dst)
        
def verboseConfusion(FN, FP, TN, TP):
    print("FALSE NEGATIVE: " + str(FN))
    print("FALSE POSITIVE: " + str(FP))
    print("TRUE NEGATIVE: " + str(TN))
    print("TRUE POSITIVE: " + str(TP))
    Accuracy = (TP + TN)/(TP+TN+FP+FN)
    Precision = TP/(TP + FN)
    Recall = TP/(TP + FP)
    
    print("Accruacy: " + str(Accuracy))
    print("Precision: " + str(Precision))
    print("Recall: " + str(Recall))

def confusionMatrixValues(eval_gt, eval_pred, imageFilePaths):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    TP_path = []
    TN_path = []
    FP_path = []
    FN_path = []
    
    for image in imageFilePaths:
        head, imageName = os.path.split(image) 
        gt = eval_gt[imageName]
        prediction = eval_pred[imageName]
        
        if gt == 0 and prediction == 0:
            TP = TP + 1
            TP_path.append(imageName)
        if gt == 1 and prediction == 1:
            TN = TN + 1
            TN_path.append(imageName)
        if gt == 0 and prediction == 1:
            FN = FN + 1
            FN_path.append(imageName)
            # cv2.imshow("false negative", imagePaths[i])
            # cv2.waitKey()
            # cv2.destroyAllWindows()
        if gt == 1 and prediction == 0:
            FP = FP + 1
            FP_path.append(imageName)
            # cv2.imshow("false positive", imagePaths[i])
            # cv2.waitKey()
            # cv2.destroyAllWindows()
    return FN, FP, TN, TP, FN_path, FP_path, TN_path, TP_path

def presentResults(eval_gt, eval_pred, imageFilePaths):
    FN, FP, TN, TP, FN_path, FP_path, TN_path, TP_path = confusionMatrixValues(eval_gt, eval_pred, imageFilePaths)
    verboseConfusion(FN, FP, TN, TP)
    return FN_path, FP_path, TN_path, TP_path

def saveResults(FN_path, FP_path, CONF_MATRIX_PATH, TEST_IMAGE_DIR, meta_data):
    import csv
    
    # CONF_MATRIX_PATH = 'D://DATA/Datasets/Detail-NonDetail/confusionMatrix/ensemble/'
    
    FN_PATH = CONF_MATRIX_PATH + 'FN/'
    for im in FN_path:
        shutil.copyfile(TEST_IMAGE_DIR + im, FN_PATH + im)
    print('FN done')
    
    FP_PATH = CONF_MATRIX_PATH + 'FP/'
    for im in FP_path:
        shutil.copyfile(TEST_IMAGE_DIR + im, FP_PATH + im)
    print('FP done')
    
    # field names  
    fields = ['Image Name', 'Classifier Prediction', 'Class', 'Classifier Score', 'Ground Truth']  
        
    # name of csv file  
    filename = "meta_data_predictions.csv"
        
    
    with open(TEST_IMAGE_DIR + filename, 'w', newline='') as f:
        thewriter = csv.DictWriter(f, fieldnames=fields)
        thewriter.writeheader()
        for meta in meta_data:
            thewriter.writerow(meta)
    print('Meta Data done')
    
def evaluateBinaryClassifier(classifier1, TEST_IMAGE_DIR, eval_gt, sort_ID, classIdx):
    
    print("starting program . . .")
    
    # Classifier 1:
    classification_c1, detection_graph_cla1 = classifier1.prepareGraphs()  
    
    sess2 = tf.Session(graph=detection_graph_cla1)
    
    # Load in file paths
    imageFilePaths, image_names = buildImageFileList(TEST_IMAGE_DIR)
    
    imageFilePaths.sort()
    cur_name = 0
    eval_pred = {}
    meta_data = []
    for image_path in imageFilePaths:
        
        head, imageName = os.path.split(image_path)
        openCVImage = cv2.imread(image_path)
        openCVImage = cv2.resize(openCVImage,(400,400))
        # if we were not able to successfully open the image, continue with the next iteration of the for loop
        if openCVImage is None:
            print("unable to open " + image_path + " as an OpenCV image")
            cur_name = cur_name + 1
            continue
        # end if 

        prediction, onMostLikelyPrediction, sortedPredictions, strClassification, confidence = makePrediction(sess2, 
                                                                                                              openCVImage, 
                                                                                                              classification_c1)
        head, imageName = os.path.split(image_path) 
        gt = eval_gt[imageName]
        #=============================================================================
        if strClassification == (sort_ID.lower()):
            score_class_conf = (confidence-.50)*2
            eval_pred.update({imageName:classIdx(strClassification)})
            meta_data.append({'Image Name':imageName, 'Classifier Prediction':0,'Class':strClassification, 
                              'Classifier Score':score_class_conf, 'Ground Truth':gt, 'Conf. Matrix':confusionMatrix(classIdx(strClassification), gt)})
        else:
            score_class_conf = (confidence-.50)*2      
            eval_pred.update({imageName:classIdx(strClassification)})  
            meta_data.append({'Image Name':imageName, 'Classifier Prediction':classIdx(strClassification),'Class':strClassification, 
                              'Classifier Score':score_class_conf, 'Ground Truth':gt, 'Conf. Matrix':confusionMatrix(classIdx(strClassification), gt)})              
        #print("==============================================================")
        cur_name = cur_name + 1  
        if (cur_name % 100) == 0:
            print ("logging: ..." + str(cur_name))          
    # end for
    return eval_pred, imageFilePaths, meta_data
# end main

def sortBinaryClassifier(classifier1, TEST_IMAGE_DIR, eval_gt, sort_ID, classIdx, BASE):
    
    print("starting program . . .")
    
    # Classifier 1:
    classification_c1, detection_graph_cla1 = classifier1.prepareGraphs()  
    
    sess2 = tf.Session(graph=detection_graph_cla1)
    
    # Load in file paths
    imageFilePaths, image_names = buildImageFileList(TEST_IMAGE_DIR)
    
    imageFilePaths.sort()
    cur_name = 0
    for image_path in imageFilePaths:
        
        head, imageName = os.path.split(image_path)
        openCVImage = cv2.imread(image_path)
        openCVImage = cv2.resize(openCVImage,(300,300))
        # if we were not able to successfully open the image, continue with the next iteration of the for loop
        if openCVImage is None:
            print("unable to open " + image_path + " as an OpenCV image")
            cur_name = cur_name + 1
            continue
        # end if 

        prediction, onMostLikelyPrediction, sortedPredictions, strClassification, confidence = makePrediction(sess2, 
                                                                                                              openCVImage, 
                                                                                                              classification_c1)
        head, imageName = os.path.split(image_path) 
        #=============================================================================
        if strClassification == (sort_ID.lower()):
            score_class_conf = (confidence-.50)*2
            sortPrediction_(image_path, BASE, score_class_conf, 
                     strClassification, imageName, sort_ID, sort_ID + "-Non" + sort_ID) 
        else:
            score_class_conf = (confidence-.50)*2      
            sortPrediction_(image_path, BASE, score_class_conf, 
                     strClassification, imageName, sort_ID, sort_ID + "-Non" + sort_ID)               
        #print("==============================================================")
        cur_name = cur_name + 1  
        if (cur_name % 100) == 0:
            print ("logging: ..." + str(cur_name))          
# end main

def binary_sort_positive_images(classifier1, SEARCH_DIRECTORY, search_ID, POSITIVE_DESINTATION_DIRECTORY):
    print("starting program . . .")
    
    # Classifier 1:
    classification_c1, detection_graph_cla1 = classifier1.prepareGraphs()  
    
    sess1 = tf.Session(graph=detection_graph_cla1)
    
    # Load in file paths
    imageFilePaths, image_names = buildImageFileList(SEARCH_DIRECTORY)
    
    imageFilePaths.sort()
    cur_name = 0
    for image_path in imageFilePaths:
        
        head, imageName = os.path.split(image_path)
        openCVImage = cv2.imread(image_path)
        openCVImage = cv2.resize(openCVImage,(300,300))
        
        # if we were not able to successfully open the image, continue with the next iteration of the for loop
        if openCVImage is None:
            print("unable to open " + image_path + " as an OpenCV image")
            cur_name = cur_name + 1
            continue
        # end if 

        prediction, onMostLikelyPrediction, sortedPredictions, strClassification, confidence = makePrediction(sess1, 
                                                                                                              openCVImage, 
                                                                                                              classification_c1)
        head, imageName = os.path.split(image_path) 
        #=============================================================================
        if strClassification == (search_ID.lower()):
            print('Found a: ' + search_ID)
            shutil.copy(image_path, POSITIVE_DESINTATION_DIRECTORY + imageName)
        cur_name = cur_name + 1  
        if (cur_name % 100) == 0:
            print ("logging: ..." + str(cur_name))          
# end main
import csv
def findHeightWidth(directory, csv_file):
    fields = ['filename', 'width', 'height']
    
    print('starting program...')
    with open(directory + csv_file, 'w', newline='') as f:
        thewriter = csv.writer(f)
        thewriter.writerow(fields)
        print('=================================================================')
        for imageFileName in os.listdir(directory):
            if imageFileName.endswith(".png"):
                print(imageFileName)
                height = cv2.imread(directory+imageFileName).shape[0]
                width = cv2.imread(directory+imageFileName).shape[1]
                thewriter.writerow([imageFileName, width, height])

# def checkAndChangeHeightandWidth(directory, csv_file):
#     print('starting program...')
#     with open(directory + csv_file, 'w', newline='') as f_read:
#         reader = csv.reader(f_read, delimiter=',')
#         with open(directory + csv_file, 'w', newline='') as f:
#             thewriter = csv.writer(f)
#             thewriter.writerow(fields)
#          for row in reader:
        
        
#         print('=================================================================')
#         for imageFileName in os.listdir(directory):
#             if imageFileName.endswith(".png"):
#                 print(imageFileName)
#                 height = cv2.imread(directory+imageFileName).shape[0]
#                 width = cv2.imread(directory+imageFileName).shape[1]
#                 thewriter.writerow([imageFileName, width, height])
        
        