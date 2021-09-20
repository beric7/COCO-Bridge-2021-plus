# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 19:49:47 2018

@author: Eric Bianchi

Developes the evaluation objects
"""

class eval_con(object):
     
    class_type = ""

    # The class "constructor" - It's actually an initializer 
    def __init__(self, name, class_type, xmin, ymin, xmax, ymax):
        self.class_type = class_type
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.name = name
    
    def toString(self):
        return(self.class_type + "," + str(self.xmin) + "," + 
               str(self.ymin) + "," + str(self.xmax) + "," + str(self.ymax))
        
    def getName(self):
        return self.name