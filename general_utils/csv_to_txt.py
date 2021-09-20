# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 23:52:17 2020

@author: Eric Bianchi
"""

import os
import csv
import sys

sys.path.insert(0, "D://Python/general_utils/")
from eval_con import eval_con

def csv_to_txt(csv_file, save_directory):

    #++++++++++++++++++++++++++++++++++++++++++
    f = open(csv_file)
    csv_f = csv.reader(f)  
    #++++++++++++++++++++++++++++++++++++++++++
    a = 0
    evalList = []
    for col in csv_f:
        if a >= 1:       
            # name, class, xmin, ymin, xmax, ymax
            info = eval_con(str(col[0]), str(col[3]), str(col[4]), str(col[5]), str(col[6]), str(col[7]))
            evalList.append(info)
        a = a + 1
        
    a = True
    repeat = ""
    eval_obj = []
    i = 0
    while i < len(evalList):
        if a == False:
            i = i + 1
        name = evalList[i].name
        
        if not os.path.exists(save_directory +"/Metrics_Eval_txt/"): # if it doesn't exist already
            os.makedirs(save_directory +"/Metrics_Eval_txt/")

        filename = save_directory +"/Metrics_Eval_txt/" + evalList[i].name + ".txt"
        f = open(filename,"w+")
        
        h = int(str(col[2]))
        w = int(str(col[1]))
        f.write(str(h) + "\n")
        f.write(str(w) + "\n")
        f.close()
        while (name == repeat or a == True) and (i < len(evalList)):
            
            #x_min = evalList[i].xmin
            #aka = evalList[i].name
            #bring back to default
            a = False
            
            if i < len(evalList) - 1:
                repeat = evalList[i+1].name  
                                
            eval_obj.append(evalList[i])          
            print(eval_obj[i].toString())           
            with open(filename, "a+") as fd:
                fd.write(eval_obj[i].toString() + "\n")  
                
            if repeat == name:    
                i = i + 1
                
            
    fd.close()
