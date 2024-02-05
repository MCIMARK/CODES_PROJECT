# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 11:45:44 2022

@author: MCIM
"""
import os
import cv2
from datetime import datetime

drive = os.getcwd()[:1]
instanteInicial = datetime.now()
datapath = drive + ':/base_de_datos/recordings_cohn_kanade'
people_list = os.listdir(datapath)
saveimag = drive + ':/base_de_datos/recordings_cohn_kanade/RESULTADOS/VIDEOS'

for address1 in people_list:#WE READ EVERY FILE OF THE ADDRESS
    people_database = datapath +'/'+address1
    if not os.path.exists(saveimag +'/'+ address1):#THE ADDRESS WHERE IT WILL BE SAVED IS CREATED
        os.makedirs(saveimag +'/'+ address1)
    for address2 in os.listdir(people_database):#WE READ EVERY FILE OF THE ADDRESS
        people_database2 = people_database+'/'+address2
        name = str(saveimag +'/'+ address1 +'/'+ address2.replace(".mp4",""))
        capture = cv2.VideoCapture(people_database2)#LOCATION OF THE CAMERA TO BE USED
        video_output = cv2.VideoWriter (str(name) + '.avi' , cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
        while (capture.isOpened()):#IF THE CAPTURE IS TRUE
            ret, image = capture.read()#READ FROM CAMERA(RET: TRUE IS FRAME IS AVAILABLE, IMAGE: FRAME)
            if ret==True:#IF RET IS TRUE
                video_output.write(image)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break                    
            else:
                break
         

            

capture.release()
video_output.release()
instanteFinal2 = datetime.now()
tiempo2 = instanteFinal2 - instanteInicial
print(tiempo2)


