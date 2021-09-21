# COCO-Bridge-2021-plus
Extension on the COCO-Bridge Dataset

[COCO-Bridge Paper]

## SSD Summary

(TODO)

## Overview
- Our dataset and model was focused on the detection of typical structual components which must be check during a bridge inspection. These components are found on the underside of steel bridges. This type of detection would be beneficial to the inspection process to aid with unmanned aerial system missions as well as localizing and contextualizing elements to semi-automate bridge inspection reports or captioning. 
  
- Training (SSD): We trained an SSD model and a YOLOv4 model. Under the same model parameters of 25% confidence and an IoU of 50%, the SSD had a  mAP score of 50% while the YOLOv4 model was able to reach 84.52%. This repository is the SSD configuration, while the YOLOv4 repository can be found here []. The drawback of the YOLOv4 model was that it was very difficult to configure, and we were only able to train and test the model in Google Colab. However, other researchers and interested people are welcome to try to get it running on your local machines!

- Testing (SSD): We implemented several testing scripts. In general we utilized a script to make predictions, and output the results in text and csv file format. Those predictions are then compared to ground truth annotations and we are able to get the mAP scores for different confidence and IoU thresholds. 

- Testing (Visualization): We can visualize and save the predictions made by the trained model, as well as view the ground truth predictions overlayed on the images. 

- Visulization: We can concatenate the results from the predictions, ground truth labels, and base images for a visual summery of the performance.  

## References

# Dataset
- The dataset information can be found [here](10.7294/16624495). **Please cite both the dataset and the journal article if you are using it.** 

# Model
- The model information can be found [here](10.7294/16625095). **Please cite both the model and the journal article if you are using it.** 

The images in the dataset were annotated using [labelme](https://github.com/wkentaro/labelme). 

The four object classes in the dataset are:
```
          Bearing
          Out of Plane Stiffener
          Gusset Plate Connection
          Cover Plate Termination:
```

## Requirements
- CMake >= 3.18
- Powershell for Windows
- Nvidia Cuda Toolkit >= 10.2
- OpenCV >= 2.4
- Nvidia cuDNN >= 8.0.2
- GPU with Compute Capability (CC) >= 3.0 (If GPU a GeForce GTX 650 or newer it is most likely compatible)

## Reproduce Results
- Clone the repository
- Download the [Dataset]()
- Download the [Trained Model]

- Download the checkpoint weight file [******](https://drive.google.com/drive/u/0/folders/1FFRRw1OoenVb4nAc70Iw1BGja1gF4SxE) to use for initial training and place within the "backup" directory.

## Setup for Training
1. Clone the [repository](https://github.com/beric7/YOLOv4_infrastructure.git), download the [Partial Dataset](https://data.lib.vt.edu/articles/dataset/COCO-Bridge_Common_Objects_in_Context_Dataset_for_Structural_Detail_Detection_of_Bridges/14097068), and download the 


2. Download the base weight file [***yolov4.conv.137***](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137) to use for initial training and place within the "darknet" directory

3. Create a directory named "obj" for the training image dataset and the image size should be a **multiple of 32 pixels**

     Example: 128x128, 320x320, 512x512 ....


4. Each image needs a corresponding txt file with the annotations for each object. Our implemenation has a script file
     xml_to_yolo_format.py that takes in each images corresponding xml file to produce the yoloV4 format txt file
     
    - Yolov4 format:
    
          <object_index> <relative_obj_x_center> <relative_obj_y_center> <relative_obj_width> <relative_obj_height>

    - Where the relative centers are the centers of each of the object's bounding boxes relative to the overall image's center
        and the relative width/height are each of the object's bounding boxes relative to the overall image's width/height

    - Example for a 320x320 image with 2 objects:
    
          Out of Plane Stiffener with <xmin>242</xmin> <ymin>23</ymin> <xmax>271</xmax> <ymax>258</ymax>

    - The label should be kept in a dictionary to maintain the a standardized <object_index> throughout.
    - In this case, "Out of Plane Stiffener" is the first object so it is assigned <object_index> = 0

          relative_obj_x_center = ((((x_max - x_min) * 1/2) + x_min)/image_width)
          relative_obj_y_center = ((((y_max - y_min) * 1/2) + y_min)/image_height)
          relative_obj_width = (x_max - x_min) / image_width
          relative_obj_height = (y_max - y_min) / image_height

    - YoloV4 Format Result: 
      
          0 0.8016 0.4391 0.09062 0.7344
       
   **Note**:
   
    - If there are 6 bounding boxes in an image, there should be a text file with 6 lines in the yolov4 format
      explained above for each image.

5. Create a ***train.txt*** file within the darknet/data directory
    - The txt file train.txt should contain the path to every image in the training dataset
      Example:
      
          data/obj/Black Camera DSCF162.jpeg
          data/obj/Black Camera DSCF164.jpeg
          data/obj/Black Camera DSCF170.jpeg
          data/obj/Black Camera DSCF182.jpeg

6. Create a ***obj.names*** and ***obj.data*** file within the darknet/data directory

    - The contents of "obj.names" should contain each labeled object in the order they are indexed above for the yolov4 format txt files
    
    - obj.names Example:
      
          Bearing
          Out of Plane Stiffener
          Gusset Plate Connection
          Cover Plate Termination

     - The contents of "obj.data" should be of the format:

           classes = <total_number_of_indexed_objects>
           train  = <path_to_train.txt>
           names = data/obj.names
           backup = backup/

7. If there is not a ***yolo-obj.cfg*** file within the cfg directory, create one and copy and paste the contents from
     yolov4-custom.cfg to the new yolo-obj.cfg

        Once the yolo-obj.cfg file is located, change the following:
          A. Ensure the width and height values on lines 8 and 9 are the correct values for you training data
          
          B. Ensure lines 6 and 7 have the following values:
            batch = 64
            subdivisions=16

          C. Set the max_batches on line 20 to the following:
            max_batches = (2000 * <total_number_of_indexed_objects>)

          D. Set the steps on line 22 to the following (make sure to include the comma below):
            steps = 0.80 * max_batches, 0.90 * max_batches

          E. Set the number of classes on lines 970, 1058, 1146 to the following:
            classes = <total_number_of_indexed_objects>
            
          F. Set the value for filters on lines 963, 1051, 1139 to the following:
            filters = (classes + 5) * 3
            
## Custom Training with Yolov4
- Download the Google Colab to Google Drive
- If not already installed, download the Google Colab App for Google Drive
- Allow Colab to provide a GPU Hardware Accelerator by clicking **edit** at the top left of the screen, then 
**Notebook Settings** and finally make sure the **GPU** dropbox is selected under **Hardware Accelerator**
- Follow the steps for **Setup for Training** and upload this directory to the user's google drive along with the images.

- Command for training if the directory was setup as shown above:
```
/darknet detector train data/obj.data cfg/yolo-obj.cfg yolov4.conv.137 -dont_show
```

- The weights are updated and saved every 1000 iterations of training and can be found within the "darknet/backup" directory.
  Training can be resumed for checkpoint weights by changing the command to the example below:
```
/darknet detector train data/obj.data cfg/yolo-obj.cfg backup/yolo-obj_3000.weights -dont_show
```

## Evaluating Trained Yolov4 Model

- The following command can be used to evaluate the model using Mean Average Precision (MAP)
```
/darknet detector map data/obj.data cfg/yolo-obj.cfg backup/yolo-obj_8000.weights
```    
    
- You can also train with the "-map" flag on the end of the above training commands to track the mAP %
  which is graphed on the chart.png within the darknet directory but requires: 'valid=valid.txt' to be added
  within the ***obj.data*** file
  
## Testing the Trained Yolov4 Model

- Once the model has been trained, the detector test should be used with the model achieving the highest mAP score. 
- The model can be tested by adding a sample image in the "darknet/data" directory and use the following command:
```
/darknet detector test data/obj.data cfg/yolo-obj.cfg backup/yolo-obj_8000.weights data/<test_image>
```

- The output image from testing will be names **prediction.jpg** and can be shown by using:
```
colabImageShow('predictions.jpg')
```

## Citation
```
hahasadf
```
