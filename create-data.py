# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 15:47:10 2025

@author: lucas
"""


import files_processing
from files_processing import *
import pandas as pd
from dataclasses import dataclass, field
import random

@dataclass
class Base :
    label1 : int = 0
    label2 : int = 0
    label3 : int = 0
    
    fora : int = 0
    forb : int = 0
    forc : int = 0
    
    label_list : list = field(default_factory=list)


b = Base()

for i in range(label.size):
    b.label_list.append(label[i])
    
b.label_list = [int(label) for label in b.label_list]




for i in range(label.size) : 
    if b.label_list[i] == 1 :
        b.label1 += 1 
    elif b.label_list[i] == 2 :
        b.label2 += 1 
    else :
        b.label3 += 1 

partie1 = np.empty((b.label1, 4 * (nb_param - 1)))
partie2 = np.empty((b.label2, 4 * (nb_param - 1)))
partie3 = np.empty((b.label3, 4 * (nb_param - 1)))


for i in range (label.size) :
    if label[i] == 1 :
        partie1[b.fora] = data[i]
        b.fora += 1
    elif label[i] == 2 :
        partie2[b.forb] = data[i]
        b.forb += 1
    else :
        partie3[b.forc] = data[i]
        b.forc += 1


partie1 = pd.DataFrame(partie1)
partie2 = pd.DataFrame(partie2)  
partie3 = pd.DataFrame(partie3)

def create_new_data(DataFrame) :
    defb = []
    defc = []
    nb_iteration = 1800
    for i in range((4 * (nb_param - 1))) :
                   defb.append(min(DataFrame[i]))
                   defb.append(max(DataFrame[i]))
                   for j in range(nb_iteration) :
                       defc.append(random.uniform(defb[0],defb[1]))
    
    
    final_data = np.array(defc).reshape(nb_iteration, (4 * (nb_param - 1)))
    
    
    label_final_data = np.zeros((nb_iteration, 1))
    label_final_data += 2
    

    return final_data, label_final_data

final_data , label_final_data = create_new_data(partie2)


data = np.concatenate((data, final_data), axis = 0)
label = np.concatenate((label, label_final_data), axis = 0)












