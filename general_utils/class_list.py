# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 20:38:42 2018

@author: Eric Bianchi
reviewed: 6/17/2020

IMPORTANT FILE. The IDs must corrospond with the IDs used to train the network

Each ID is assigned to a specific class. EVERY time you update the number of classes, 
you MUST adjust accordingly. 
"""

class class_list(object):

    # The class "constructor" - It's actually an initializer 
    def __init__(self, ID, ID_list):
        if ID_list == "Structural Details":
            if ID == 1:
                self.ID = "Gusset Plate Connection"
            elif ID == 2:
                self.ID = "Out of Plane Stiffener"
            elif ID == 3:
                self.ID = "Cover Plate Termination"
            elif ID == 4:
                self.ID = "Bearing"
                
        if ID_list == "Structural Cracks":
            if ID == 1:
                self.ID = "Crack Damage"
            elif ID == 2:
                self.ID = "Crack Damage"
            elif ID == 3:
                self.ID = "Crack Damage"
            elif ID == 4:
                self.ID = "Crack Damage"
                
        if ID_list == "Rating":
            if ID == 1:
                self.ID = "Severe (4)"
            elif ID == 2:
                self.ID = "Poor (3)"
            elif ID == 3:
                self.ID = "Fair (2)"
            elif ID == 4:
                self.ID = "Good (1)" 
        
        self.ID_list = ID_list
            


            
