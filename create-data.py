# import des librairies

import files_processing
from files_processing import *
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
import pandas as pd

# séparation des différents labels en trois catégories (1 ,2 et 3)

label_list = []
for i in range(label.size):
    label_list.append(label[i])
    
label_list = [int(label) for label in label_list]

label1 = 0
label2 = 0
label3 = 0

for i in range(label.size) : 
    if label_list[i] == 1 :
        label1 += 1 
    elif label_list[i] == 2 :
        label2 += 1 
    else :
        label3 += 1 

# répartition des données en fonction de leur label 

partie1 = np.empty((label1, 4 * (nb_param - 1)))
partie2 = np.empty((label2, 4 * (nb_param - 1)))
partie3 = np.empty((label3, 4 * (nb_param - 1)))

a = 0
b = 0
c = 0

for i in range (label.size) :
    if label[i] == 1 :
        partie1[a] = data[i]
        a += 1
    elif label[i] == 2 :
        partie2[b] = data[i]
        b += 1
    else :
        partie3[c] = data[i]
        c += 1

# création des metadatas pour la création de données
title = [None] * (4 * (nb_param - 1))
for i in range(13) :
    title[i*4] = ('Amplitude')
    title[i*4 + 1] = ('TOF')
    title[i*4 + 2] = ('TOF_Peak')
    title[i*4 + 3] = ('Temperature')
    
title = pd.DataFrame([title])


meta = {}
meta[f"metadata"] = title


partie1 = pd.DataFrame(partie1)
partie2 = pd.DataFrame(partie2)  
partie3 = pd.DataFrame(partie3)

# création de données detype label 2
data_test = {}
data_test[f"data_test"] = []

synthetic_data = {}
synthetic_data[f"synthetic_data"] = []

data_test[f'data_test'] = partie2
data_test[f'data_test'] = pd.concat([meta[f'metadata'] , data_test[f'data_test']], axis = 0)
metadata = Metadata.detect_from_dataframe(data=meta[f'metadata'])

model = CTGANSynthesizer(metadata=metadata, epochs=30)

model.fit(data_test[f'data_test'])
nb_iteration= 300

synthetic_data[f"synthetic_data"] = model.sample(nb_iteration)
    
# synthetic_data[f"synthetic_data1"].iloc[:, 0] = synthetic_data[f"synthetic_data1"].iloc[:, 1]
# synthetic_data[f"synthetic_data2"].iloc[:, 0] = synthetic_data[f"synthetic_data2"].iloc[:, 1]
# synthetic_data[f"synthetic_data3"].iloc[:, 0] = synthetic_data[f"synthetic_data3"].iloc[:, 1]
# synthetic_data[f"synthetic_data4"].iloc[:, 1] = synthetic_data[f"synthetic_data4"].iloc[:, 0]

synthetic_data[f"synthetic_data"] = synthetic_data[f"synthetic_data"].to_numpy()
# synthetic_data[f"synthetic_data2"] = synthetic_data[f"synthetic_data2"].to_numpy()
# synthetic_data[f"synthetic_data3"] = synthetic_data[f"synthetic_data3"].to_numpy()
# synthetic_data[f"synthetic_data4"] = synthetic_data[f"synthetic_data4"].to_numpy()

# fonction pour vérifier si une valeur est un float ou non

def is_float(value):
    try:
        return isinstance(value, float)
    except (ValueError, TypeError):
        return False

# vérification du type des données crées, si elles sont des float, les remplaces par la valeur précédente
for j in range(synthetic_data[f"synthetic_data"].shape[1]):
    # Variable pour stocker la dernière valeur non float
    last_valid_value = None
    
    for i in range(synthetic_data[f"synthetic_data"].shape[0]):
        if is_float(synthetic_data[f"synthetic_data"][i, j]):
            if last_valid_value is not None:
                synthetic_data[f"synthetic_data"][i, j] = last_valid_value
        else:
            last_valid_value = synthetic_data[f"synthetic_data"][i, j]

# final_data = np.zeros((nb_iteration, 4 * (nb_param - 1)))
# for i in range(nb_iteration) :
#     final_data[i] = np.concatenate((synthetic_data[f"synthetic_data1"][i], synthetic_data[f"synthetic_data2"][i], synthetic_data[f"synthetic_data3"][i], synthetic_data[f"synthetic_data4"][i]))


label_final_data = np.zeros((nb_iteration, 1))
label_final_data += 2

# réunion des données de base et des données crées

data = np.concatenate((data, final_data), axis = 0)
label = np.concatenate((label, label_final_data), axis = 0)
