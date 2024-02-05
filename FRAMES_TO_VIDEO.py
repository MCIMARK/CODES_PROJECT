# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 12:11:46 2022

@author: MCIM
"""
#WE IMPORT BOOKSTORES AND PACKAGE STORES TO OCCUPY
import os
import cv2

datapath = 'D:/base_de_datos/database_full/CK database/cohn-kanade-images'
people_list = os.listdir(datapath)
save_video = 'D:/base_de_datos/database_full/CK database/cohn-kanade-video'


if not os.path.exists(save_video):#THE ADDRESS WHERE IT WILL BE SAVED IS CREATED
    os.makedirs(save_video)
for address1 in people_list:#WE READ EVERY FILE OF THE ADDRESS
    people_database = datapath +'/'+address1
    if not os.path.exists(save_video + '/' + address1):#THE ADDRESS WHERE IT WILL BE SAVED IS CREATED
        os.makedirs(save_video + '/' + address1)
    for address2 in os.listdir(people_database):#WE READ EVERY FILE OF THE ADDRESS
        people_database2 = people_database +'/' + address2
        video_output = cv2.VideoWriter (str(save_video + '/' + address1+ '/' + address2) + '.avi' , cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
        for address3 in os.listdir(people_database2):#WE READ EVERY FILE OF THE ADDRESS
            people_database3 = people_database2 +'/' + address3
            imagen = cv2.imread(people_database3)
            canva = cv2.resize(imagen,(640,480),interpolation=cv2.INTER_CUBIC)
            cv2.imshow('imagen',canva)
            video_output.write(canva)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # else:
            #     break
video_output.release()
cv2.destroyAllWindows()