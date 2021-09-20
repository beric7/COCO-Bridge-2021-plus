# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:39:06 2020

@author: Eric Bianchi
"""

import csv

def dict_to_csv(dictionary, fields, destination):
    
    with open(destination, 'w', newline='') as f:
        thewriter = csv.DictWriter(f, fieldnames=fields)
        thewriter.writeheader()
        for index in dictionary:
            thewriter.writerow(index)
        print('done.')
        
def array_to_csv(array, fields, destination):
    
    with open(destination, 'w', newline='') as f:
        thewriter = csv.writer(f)
        thewriter.writerow(fields) 
        for i in range (0,len(array)):
            print(array[i])
            thewriter.writerow([array[i]])
        print('done.')