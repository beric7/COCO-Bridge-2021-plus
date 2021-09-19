# COCO-Bridge-2021-plus
Extension on the COCO-Bridge Dataset

[COCO-Bridge Paper]



## SSD Summary

- YoloV4 is a real-time state-of-the-arc object detector. Modern Neural Networks operated in real-time
  require significant power from multiple GPU's, while YoloV4 uses a Convolutional Neural Network (CNN) that
  can reduce the consumption to one singular GPU. YoloV4 has comparable results to state-of-the-art real-time
  object detection models and runs twice as fast

## Overview
- Our implementation had difficulties running from console due to compatibility issues with cmake
  and opencv while trying to build the project, but had promising results by implementing Yolov4
  through [Google Colab](https://colab.research.google.com/drive/1MOjUxrYAl05Be2QVPyDC0D8bZ2TVXCif) 
  
- Our dataset was focused on the structual components found on the underside of bridges. The images in the dataset were annotated
  using [labelme](https://github.com/wkentaro/labelme). 
  
- Created a conversion script, **keypoint-to-text-to-json.py**, to take the resultant labelme json file and convert it to text file format so the keypoint values could
  be stored and used to create the xml format annotations for our objects. This script then converts back to json format.

- Created a yolov4 conversion script, **xml_to_yolo_format.py**, to convert our xml annotations to the yolov4 textfile format

- The model achieved the highest **Mean Average Precision (mAP)** score at the **7000 iteration** checkpoint with a **mAP score of 84.52%**
  when detecting these 4 structual component:
```
          Bearing
          Out of Plane Stiffener
          Gusset Plate Connection
          Cover Plate Termination:
```
- The following results are similarity scores for the model's predicted bounding boxes and the annotated bounding boxes:
```
Gusset Plate Connection: 99%	(left_x:    0   top_y:  214   width:   53   height:   66)
Out of Plane Stiffener: 97%	(left_x:   51   top_y:  152   width:   21   height:  163)
Out of Plane Stiffener: 95%	(left_x:   99   top_y:  118   width:   19   height:  192)
Gusset Plate Connection: 100%	(left_x:  150   top_y:   67   width:   38   height:   82)
Gusset Plate Connection: 95%	(left_x:  153   top_y:  252   width:   35   height:   31)
Out of Plane Stiffener: 99%	(left_x:  166   top_y:   59   width:   28   height:  253)
Out of Plane Stiffener: 94%	(left_x:  278   top_y:   -1   width:   43   height:  307)
```
- The model began to experience overfitting around **8000 iterations** and therefore the mAP score would decrease with further iterations

<p align="center">
    <img src="https://user-images.githubusercontent.com/54971419/124635837-e45fc480-de55-11eb-97ea-d66ea55fcf8b.png" />
    <img src="https://user-images.githubusercontent.com/54971419/124629491-7e703e80-de4f-11eb-9b9a-08bc2cb14d09.png" />
</p>

<p align="center">
    <img src="https://user-images.githubusercontent.com/54971419/124635758-cb571380-de55-11eb-8d3d-419603c25c81.png" />
    <img src="https://user-images.githubusercontent.com/54971419/124635122-145a9800-de55-11eb-844f-230ba745c5aa.png" />
</p>

<p align="center">
    <img src="https://user-images.githubusercontent.com/54971419/125976914-07381084-6c3b-4ba1-b7e8-ce53e3bd436f.png" />
    <img src="https://user-images.githubusercontent.com/54971419/125977251-140ac061-57fa-4f58-8230-dd287f179fe5.png" />
</p>

<p align="center">
    <img src="https://user-images.githubusercontent.com/54971419/125977387-fac0a79d-369f-4c27-a577-56bb82e6937d.png" />
    <img src="https://user-images.githubusercontent.com/54971419/125977460-54fe4954-bd71-40b9-896e-b9af1e2daf7b.png" />
</p>

## Requirements
- CMake >= 3.18
- Powershell for Windows
- Nvidia Cuda Toolkit >= 10.2
- OpenCV >= 2.4
- Nvidia cuDNN >= 8.0.2
- GPU with Compute Capability (CC) >= 3.0 (If GPU a GeForce GTX 650 or newer it is most likely compatible)

## Reproduce Results
- Download the [Google Colab](https://drive.google.com/drive/u/1/folders/1FFRRw1OoenVb4nAc70Iw1BGja1gF4SxE) to Google Drive
  
- Clone the [repository](https://github.com/beric7/YOLOv4_infrastructure.git) and download the [Partial Dataset](https://data.lib.vt.edu/articles/dataset/COCO-Bridge_Common_Objects_in_Context_Dataset_for_Structural_Detail_Detection_of_Bridges/14097068) to use for detection on the pretrained model.

- Download the checkpoint weight file [***yolo-obj_7000.weights***](https://drive.google.com/drive/u/0/folders/1FFRRw1OoenVb4nAc70Iw1BGja1gF4SxE) to use for initial training and place within the "backup" directory.

## Setup for Training
1. Clone the [repository](https://github.com/beric7/YOLOv4_infrastructure.git), download the [Partial Dataset](https://data.lib.vt.edu/articles/dataset/COCO-Bridge_Common_Objects_in_Context_Dataset_for_Structural_Detail_Detection_of_Bridges/14097068), and download the [google colab](https://colab.research.google.com/drive/1MOjUxrYAl05Be2QVPyDC0D8bZ2TVXCif)


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
