Python Scraping Package

Author: Eric Bianchi
Date: 3/17/2020 - 11:50AM

+++++++++++++++++++++++++++++++++++++++++++

=========================================================================
FOLDER:

utils : data visualization
object_detection : data_visualization

=========================================================================
CLASS:

class_list.py : Lists all objects being classified.
-- __init__(self, ID, ID_list)
	@param: ID_list = class domain
	@param: ID = class ID number
=========================================================================
CLASS:

eval_bbox.py : Created an object to record bounding box characteristics.
-- __init__(self, ID, class_type, score, xmin, ymin, xmax, ymax)
	@param: ID = class_list ID
	@param: class_type = which class was predicted
	@param: xmin, xmax, ymin, ymax = bounds of box
	@param: score = confidence

--toString(self)
    return: self.class_type + "," + str(self.score) + "," + str(self.xmin) + "," + 
               str(self.ymin) + "," + str(self.xmax) + "," + str(self.ymax)

=========================================================================
ex2image.py : Specific helper functions to extract images froms docx files. 
-- scrape_images(directory, extension, district, leader1, provider)
	@param: directory = path to directory
	@param: extension = checks the extension (docx)
	@param: district = district of image
	@param: leader1 = Directory
	@param: provider = provider of data, i.e VDOT.

-- scrape_helper(PATH, counter, extension, name, district, leader1, provider)
	@param: PATH = path to directory
	@param: counter = count of image in file
	@param: extension = type of file that it is reading
	@param: name = name of file
	@param: district = district
	@param: leader1 = Directory
	@param: provider = provider of data, i.e VDOT.
	
	return: count = number of images extracted

=========================================================================
Model.py : Loads in a model from model location in drive
-- __init__(self, Model, threshold, graph_name, pwd, color)
	@param: Model = the name of the model being used
	@param: threshold = the threshold of confidence being used on the model
	@param: graph_name = type
	@param: pwd = location of the model directory (before /Pre-Processing)
	@param: color = color for the bounding box

-- thresholdToString(self)
    return: str(int(self.threshold*100)) + "%" = Confidence threshold

-- loadGraph(self)
    return: detection_graph

-- getDict(self)
    return: dictionary = This is a dictionary of the assigned detection number 
                            to its value. 
    
=========================================================================
test_bbox_within_bbox.py : Bounding Box within Bounding Box reporting
	Tensorflow 1.10.0
	Python 3.6.0
-- main(cur, model)
	@param: cur = directory
	@param: model = list of models which are being compared

-- analyzeIm(model, cur, image_names, imageFilePaths)
    // Can show predictions on image, and writes data to a .txt file.
	@param: model = list of models which are being compared
	@param: cur = directory
	@param: image_names = list of image names
	@param: imageFilePaths = list of paths to images

-- intersection(detect_object_list)
    // Compares the other models against a base model, i.e Structural Bridge Details
	@param: detect_object_list = list of detected objects in the image

-- compareBBOX(cur_bbox, detected_object_list)
    // Compares each bounding box for a given model against each base model bounding box
    // and checks for IOU threshold overlap. 
	@param: cur_bbox = current bounding box being compared to object of interest
	@param: detected_object_list = list of detected objects on given image

-- compareIOU(curr_bbox, compare_bbox, ovmax)
    // This is a helper method for the compareBBOX. This is where the math is done for the 
    // overlaps. It outputs some print() information for the user while running.
	@param: curr_box = current bounding box being compared to object of interest
	@param: compare_bbox = bounding box that is being compared against. 

-- modelPredictionImg(mod, image_np, scores_list, boxes_list, classes_list, cat_index_list)
    // This is for outputting a visual image with all the predictions on the image
    // from each of the models, super-imposed on eachother.
    @param: mod = model
    @param: image_np = numpy image
    @param: scores_list = list of scores for predicted bounding boxes
    @param: boxes_list = a list of bounding box arrays for each model
    @param: class_list = a list of class array for each model 
    @param: cat_index_list = a list of numbers for each class in a model
    
    return: scores_list, boxes_list, classes_list, cat_index_list 

-- findBoxCoord(scores, boxes, classes, h, w, image_names, cur_name, mod, f)
    // This method finds all potential bounding boxes for a given image across
    // across all models.
    @param: scores = an array of scores for a given model
    @param: boxes = an array of detected bounding boxes for a given model
    @param: classes_list = a list of classs for a given model
    @param: h = height of image
    @param: w = width of image
    @param: image_names = image name
    @param: cur_name = current file name
    @param: mod = model
    @param: f = file 
    
    return: f, detect_obj

-- tfVisualization(model, image_np, scores_list, boxes_list, classes_list, cat_index_list)
    // This is a part of the visualization process of the output on the image and 
    // it is important to do this way since it is looping over several models. 
    @param: mod = model
    @param: image_np = numpy image
    @param: scores_list = list of scores for predicted bounding boxes
    @param: boxes_list = a list of bounding box arrays for each model
    @param: class_list = a list of class array for each model 
    @param: cat_index_list = a list of numbers for each class in a model
=========================================================================
TXT_to_XML.py : converts textfile input into CSV output.
    Python 3.6.0
-- txt2xml (filename, file_path, folder_dir, txt_file_path, thres_str, current_dir)
    @param: filename = the name of the file input
    @param: file_path = the full file path
    @param: folder_dir = the parent folder of the file
    @param: txt_file_path = the file path for the output of the txtFile
    @param: thres_str = the threshold of the prediction on the image
    @param: current_dir = the current directory being operated in (E://, C:// etc.)
    
-- prettify(elem)
    @param: elem = xml element
    return: an xml file with all the nice indentation etc.
=========================================================================

















