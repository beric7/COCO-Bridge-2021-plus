# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 09:27:18 2020

@author: Eric Bianchi
"""
import sys
import os
from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
from xml.dom import minidom

sys.path.insert(0, 'D://Python/general_utils/')
from csv_to_txt import csv_to_txt

database = "(Bianchi, E. (2018).COCO-Bridge [Dataset]. University Libraries, Virginia Tech)"
segmented = 0

def txt2xml(filename_ext, filename, txt_file_path, save_directory):
    # filename = just the name of the file
    # file_path = the full file path
    # folder_dir = is just the parent folder of the file
    root = Element('annotation')
    
    folder = SubElement(root, 'folder')
    folder.text = 'xml'
    
    file_name = SubElement(root, 'filename')
    file_name.text = filename_ext
    
    path = SubElement(root, 'path')
    path.text = save_directory
    
    source = SubElement(root, 'source')
    DB = SubElement(source, 'database')
    DB.text = database
    
    txt_file_path = txt_file_path.replace(os.sep, '/')
    
    f = open(txt_file_path, 'r')

    size = SubElement(root, 'size')
    
    width = SubElement(size, 'width')
    line = f.readline()
    width.text = str(line)
    height = SubElement(size, 'height')
    line = f.readline()
    height.text = str(line)
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
       xmin_info = info[1]
       ymin_info = info[2]
       xmax_info = info[3]
       ymax_info = info[4]
       
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
       
    
    # print (prettify(root))
    f.close()
    
    if not os.path.exists(save_directory): # if it doesn't exist already
        os.makedirs(save_directory)
    
    open_file = save_directory + filename + ".xml"                
    f_xml = open(open_file,"w+")
    f_xml.write(prettify(root))
    f_xml.close()
    

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


# save_directory + "/Metrics_Eval_txt/"
def csv_to_xml(csv_file, save_directory_txt, save_directory_xml):
    
    if not os.path.exists(save_directory_txt): # if it doesn't exist already
        os.makedirs(save_directory_txt)
    
    csv_to_txt(csv_file, save_directory_txt)
    
    for txt_file_path in os.listdir(save_directory_txt + '/Metrics_Eval_txt/'):
        
        filename, extension_image, extension_txt = txt_file_path.split(".")
        txt_file_path = save_directory_txt + '/Metrics_Eval_txt/' + txt_file_path
        
        txt2xml(filename+'.'+extension_image, filename, txt_file_path, save_directory_xml)
        