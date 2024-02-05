# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 10:55:39 2022

@author: MCIM
"""

import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import  OneHotEncoder

drive = os.getcwd()[:1]
datapath = drive + ':/base_de_datos/full_database_project/RESULTADOS/EXCELS'
savenpy = drive + ':/base_de_datos/full_database_project/RESULTADOS/EXCELS/'
label = 0
name = ['frames', 'horizontal_lip', 'vertical_lip', 
        'right_eye', 'left_eye', 'right_brow', 
        'left_brow', 'chin', 'between_eyebrows_1', 
        'between_eyebrows_2', 'bridge_1', 'bridge_2',
        'bridge_3', 'bridge_4']
total = np.empty((0,13,180),float)
clases = []

for address in os.listdir(datapath):#WE READ EVERY FILE OF THE ADDRESS
    people_database = datapath +'/'+ address
    label = 0
    for address2 in os.listdir(people_database):#WE READ EVERY FILE OF THE ADDRESS
        archivo = people_database +'/'+address2
        df = pd.read_excel(archivo)
        fil,col = df.shape
        while (fil <= 179):
            frames = [df,df]
            df = pd.concat(frames)
            fil,col = df.shape
        if label <=179:
            label = label + 1
            clases.append(address)
            nuevo = df.iloc[:180,1:14].T
            nuevo2 = nuevo.to_numpy()
            nuevo2 = nuevo2[np.newaxis]
            total = np.append(total,nuevo2,axis=0)
clases = np.array(clases)
xdim = clases.reshape(-1, 1)
enc = OneHotEncoder()
onehotlabels = enc.fit_transform(xdim).toarray()
clases = onehotlabels[:,:4]#se modifica dependiendo las clases         
datos = np.save(savenpy + 'dat.npy',total)
clas = np.save(savenpy + 'lab.npy',clases)
caracteristicas = np.save(savenpy + 'features.npy',name)
