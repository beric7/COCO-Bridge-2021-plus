# -*- coding: utf-8 -*-
"""
Created on Fri May  1 18:38:20 2020

@author: Eric Bianchi
"""

class evalObject(object):
     
    class_type = ""
    coord = []

    # The class "constructor" - It's actually an initializer 
    def __init__(self, className, prediction, model, score, filename):
        self.prediction = prediction
        self.score = score
        self.className = className
        self.model = model
        self.filename = filename
    
    def toString(self):
        return(self.className + "," + str(self.score))
        