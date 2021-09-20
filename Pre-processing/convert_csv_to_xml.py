# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 10:09:27 2020

@author: Eric Bianchi
"""

import sys
sys.path.insert(0, '../general_utils/')
from csv_to_xml import csv_to_xml


csv_file = './Train/320x320/bbox/csv/Bbox_info_CSV_output_normal_320.csv'
save_directory_txt = './Train/320x320/bbox/txt/'

save_directory_xml = './Train/320x320/bbox/xml/'

csv_to_xml(csv_file, save_directory_txt, save_directory_xml)
