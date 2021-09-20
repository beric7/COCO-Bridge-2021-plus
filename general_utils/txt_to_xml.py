# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:27:53 2019

@author: Eric Bianchi
reviewed: 6/17/2020
"""

from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
from xml.dom import minidom
import os

#============================================================================
# Set variables
#============================================================================


database = "(Bianchi, E. (2018).COCO-Bridge [Dataset]. University Libraries, Virginia Tech)"
segmented = 0

def txt2xml (filename, file_path, folder_dir, txt_file_path, thres_str, current_dir):
    # filename = just the name of the file
    # file_path = the full file path
    # folder_dir = is just the parent folder of the file
    root = Element('annotation')
    
    folder = SubElement(root, 'folder')
    folder.text = folder_dir
    
    file_name = SubElement(root, 'filename')
    file_name.text = filename
    
    path = SubElement(root, 'path')
    path.text = file_path
    
    source = SubElement(root, 'source')
    DB = SubElement(source, 'database')
    DB.text = database
    
    txt_file_path = txt_file_path.replace(os.sep, '/')
    
    f = open(txt_file_path, 'r')

    size = SubElement(root, 'size')
    
    height = SubElement(size, 'height')
    line = f.readline()
    height.text = str(line)
    width = SubElement(size, 'width')
    line = f.readline()
    width.text = str(line)
    depth = SubElement(size, 'depth')
    depth.text = str(3)
    
    folder = SubElement(root, 'segmented')
    folder.text = str(segmented)
    
    """
    
    OBJECT
    
    """
        
    
    while line:
       
       line = f.readline()
       info = line.split(",")
       if len(line) < 1:
           break
       obj = SubElement(root, 'object')
       className = info[0]
#      score = info[1]
       xmin_info = info[2]
       ymin_info = info[3]
       xmax_info = info[4]
       ymax_info = info[5]
       
       name = SubElement(obj, 'name')
       name.text = className
       
       pose = SubElement(obj, 'pose')
       pose.text = 'Unspecified'      
       truncated = SubElement(obj, 'truncated')
       truncated.text = str(0)
       difficult = SubElement(obj, 'difficult')
       difficult.text = str(0)   
       
       bbox = SubElement(obj, 'bndbox')
       xmin = SubElement(bbox, 'xmin')
       xmin.text = str(xmin_info)
       ymin = SubElement(bbox, 'ymin')
       ymin.text = str(ymin_info)
       xmax = SubElement(bbox, 'xmax')
       xmax.text = str(xmax_info)
       ymax = SubElement(bbox, 'ymax')
       ymax.text = str(ymax_info)
       
    
    print (prettify(root))
    f.close()
    
    (head, tail) = filename.split(".")
    currentName = head
    open_file = current_dir + "Threshold_%/" + thres_str + "_Metrics_Detect_xml/" + currentName + ".xml"                
    f_xml = open(open_file,"w+")
    f_xml.write(prettify(root))
    f_xml.close()
    

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")
   
        

    