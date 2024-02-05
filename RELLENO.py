# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:08:39 2023

@author: MARCOS
"""
import os
import random
import glob
import shutil

# drive = os.getcwd()[:1]

datapath = 'D:/base_de_datos/full_database_project/RESULTADOS/EXCELS'

label = 0

for address in os.listdir(datapath):#WE READ EVERY FILE OF THE ADDRESS
    people_database = datapath +'/'+ address
    label = 0
    Numero=len(glob.glob(people_database+"/*.xlsx"))
    Numero
    for n in range(450-Numero):
        p = random.randrange(1,Numero)
        print(f'numero random: {p}')
        copia = n+Numero+1
        print(f'numero de copia: {copia}')
        fuente = people_database + '/' + str(p) +".xlsx"
        destino = people_database + '/' + str(copia) + '.xlsx'
        shutil.copyfile(fuente, destino)
        # copia = random.randrange(Numero)
    # for address2 in os.listdir(people_database):#WE READ EVERY FILE OF THE ADDRESS
    #     archivo = people_database +'/'+address2
    #     numero = np.size(people_database)
        # df = pd.read_excel(archivo)
        # random.randrange(stop)