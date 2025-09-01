# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 11:45:56 2022

@author: lucas
"""

#import des librairies
import scipy.io as sc
import numpy as np
from sklearn.model_selection import train_test_split

#récupération des fichiers de données
Loadmat = 'C:/Users/lucas/.spyder-py3/ML/'
amplitude_mat = sc.loadmat(Loadmat + 'params_amplitude_BDD.mat')
temperature_mat = sc.loadmat(Loadmat + 'params_temperature_BDD.mat')
tof_peak_mat = sc.loadmat(Loadmat + 'params_TOFpeak_BDD.mat')
tof_mat = sc.loadmat(Loadmat + 'params_TOF_BDD.mat')

amplitude_data = amplitude_mat['params']  
temperature_data = temperature_mat['params']
tof_peak_data = tof_peak_mat['params']
tof_data = tof_mat['params']

#récupération des tailles des fichiers
nb_signal = amplitude_data.shape[0]
nb_element = amplitude_data.shape[1]
nb_param = amplitude_data.shape[2]

#remise en forme des fichiers de données
amplitude_data = np.split(amplitude_data, indices_or_sections = nb_signal, axis = 0)
temperature_data = np.split(temperature_data, indices_or_sections = nb_signal, axis = 0)
tof_peak_data = np.split(tof_peak_data, indices_or_sections = nb_signal, axis = 0)
tof_data = np.split(tof_data, indices_or_sections = nb_signal, axis = 0)

#initialisation des tableaux data et label
data = np.zeros((nb_signal, nb_element, 4 * (nb_param - 1)))
label = np.zeros((nb_signal, nb_element))

#fonction nous permettant de séparer le label du reste des données
def split_last_col(array):
    """returns a tuple of two matrices corresponding 
    to the Left and Right parts"""
    A = [line[:-1] for line in array]
    B = [line[-1] for line in array]
    A = np.stack(A)
    B = np.stack(B)
    B = np.ravel(B)
    return A, B

#cette boucle nous permet de séparer le label du reste des données
#grâce à la fonction split_last_col
for i in range(0, nb_signal):
    amplitude_data[i] = amplitude_data[i].reshape(amplitude_data[i].shape[0]*amplitude_data[i].shape[1], amplitude_data[i].shape[2])
    amplitude_data[i], label[i] = split_last_col(amplitude_data[i])
    tof_data[i] = tof_data[i].reshape(tof_data[i].shape[0]*tof_data[i].shape[1], tof_data[i].shape[2])
    tof_data[i], label[i] = split_last_col(tof_data[i])
    tof_peak_data[i] = tof_peak_data[i].reshape(tof_peak_data[i].shape[0]*tof_peak_data[i].shape[1], tof_peak_data[i].shape[2])
    tof_peak_data[i], label[i] = split_last_col(tof_peak_data[i])
    temperature_data[i] = temperature_data[i].reshape(temperature_data[i].shape[0]*temperature_data[i].shape[1], temperature_data[i].shape[2])
    temperature_data[i], label[i] = split_last_col(temperature_data[i])

#implémentation du tableaux data avec les données
    data[i] = np.concatenate((amplitude_data[i], tof_data[i], tof_peak_data[i], temperature_data[i]), axis = 1)

#remise en forme des tableaux data et label
data = np.reshape(data, (nb_signal * nb_element, 4 * (nb_param - 1 )))
label = np.reshape(label, (nb_signal * nb_element, 1))











