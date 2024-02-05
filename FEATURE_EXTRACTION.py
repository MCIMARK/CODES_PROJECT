# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 11:45:44 2022

@author: MCIM
"""
#WE IMPORT BOOKSTORES AND PACKAGE STORES TO OCCUPY
import os
import cv2
import mediapipe as mp
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from pandas import ExcelWriter

drive = "C:/Users/MARCOS/Documents"
#WE DECLARE VARIABLES TO OCCUPY
instanteInicial = datetime.now()#PROGRAM START TIME
face_detection = mp.solutions.face_detection#FACE DETECTION UTILITY
mp_drawing = mp.solutions.drawing_utils#MEDIAPIPE DRAWING UTILITIES
conf_drawing = mp_drawing.DrawingSpec(color = (0,0,0),thickness=1,circle_radius=1)
t = 2 #LINE WIDTH
datapath = drive + '/PARA_PRUEBA'
people_list = os.listdir(datapath)
# saveimag = drive + ':/base_de_datos/recordings_cohn_kanade/RESULTADOS/VECTORES'
graficos = drive + '/GRAFICAS2'
excel = drive + '/EXCELS3'
# excel = drive + '/PARA_PRUEBA/EXCELS2'
# otras = drive + ':/base_de_datos/recordings_cohn_kanade/RESULTADOS/FRAMES'
mp_face_mesh= mp.solutions.face_mesh#MEDIAPIPE FACIAL MESH SOLUTION
face_mesh= mp_face_mesh.FaceMesh(static_image_mode=True,#FALSE IF VIDEO, TRUE IF IMAGE
                                 max_num_faces=1)#NUMBER OF FACES TO DETECT
with face_detection.FaceDetection(model_selection=1,#RANGE FOR FACES AT 5 METERS, 0 FOR FACES AT 2 METERS
                                    min_detection_confidence=0.5) as face_detection:#TRUST VALUE
    # if not os.path.exists(saveimag):#THE ADDRESS WHERE IT WILL BE SAVED IS CREATED
    #     os.makedirs(saveimag)
    if not os.path.exists(graficos):#THE ADDRESS WHERE IT WILL BE SAVED IS CREATED
        os.makedirs(graficos)
    if not os.path.exists(excel):#THE ADDRESS WHERE IT WILL BE SAVED IS CREATED
        os.makedirs(excel)
    # if not os.path.exists(otras):#THE ADDRESS WHERE IT WILL BE SAVED IS CREATED
    #     os.makedirs(otras)
    for address1 in people_list:#WE READ EVERY FILE OF THE ADDRESS
        people_database = datapath +'/'+address1
        cont = 0
        # if not os.path.exists(saveimag +'/'+ address1):#THE ADDRESS WHERE IT WILL BE SAVED IS CREATED
        #     os.makedirs(saveimag +'/'+ address1)
        if not os.path.exists(graficos +'/'+ address1):#THE ADDRESS WHERE IT WILL BE SAVED IS CREATED
            os.makedirs(graficos +'/'+ address1)
        if not os.path.exists(excel +'/'+ address1):#THE ADDRESS WHERE IT WILL BE SAVED IS CREATED
            os.makedirs(excel +'/'+ address1)
        # if not os.path.exists(otras+'/'+ address1):#THE ADDRESS WHERE IT WILL BE SAVED IS CREATED
        #     os.makedirs(otras +'/'+ address1)
        for address2 in os.listdir(people_database):#WE READ EVERY FILE OF THE ADDRESS
            people_database2 = people_database+'/'+address2
            # name = str(saveimag +'/'+ address1 +'/'+ address2.replace(".mp4",""))
            # print(people_database2)
            horizontal_lip, vertical_lip, right_eye, left_eye, right_brow = [], [], [], [], []
            left_brow, chin, between_eyebrows_1, between_eyebrows_2 = [], [], [], []
            bridges_1, bridges_2, bridges_3, bridges_4, frames, clase = [], [], [], [], [], []
            contador = []#COUNTER
            points = []#POINTS
            label = 0
            cont = cont + 1
            capture = cv2.VideoCapture(people_database2)#LOCATION OF THE CAMERA TO BE USED
            grafica = str(graficos +'/'+ address1 +'/'+ address2.replace(".avi","_"))
            # frame = str(otras +'/'+ address1 +'/'+ address2.replace(".avi","_"))
            datos_excel = str(excel +'/'+ address1 +'/'+ str(cont))
            # video_output = cv2.VideoWriter (str(name) + '_P.avi' , cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
            # video_output2 = cv2.VideoWriter (str(name) + '_G.avi' , cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
            # video_output3 = cv2.VideoWriter (str(name) + '.avi' , cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
            while (capture.isOpened()):#IF THE CAPTURE IS TRUE
                ret, image = capture.read()#READ FROM CAMERA(RET: TRUE IS FRAME IS AVAILABLE, IMAGE: FRAME)
                if ret==True:#IF RET IS TRUE
                    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))#FACE DETECTION IN IMAGE
                    height = image.shape[0]#IMAGE HEIGHT
                    width = image.shape[1]#IMAGE WIDTH
                    if results.detections is not None:
                        for detection in results.detections:
                            xmin = int(detection.location_data.relative_bounding_box.xmin * width)#MINIMUM VALUE OF THE IMAGE BOX IN X
                            ymin = int(detection.location_data.relative_bounding_box.ymin * height)#MINIMUM VALUE OF THE IMAGE BOX IN Y
                            w = int(detection.location_data.relative_bounding_box.width * width)#IMAGE BOX WIDTH VALUE
                            h = int(detection.location_data.relative_bounding_box.height * height)#IMAGE BOX HEIGHT VALUE
                            new_image = image[ymin : ymin + h, xmin : xmin + w]#CROPPED IMAGE
                            rectangle = cv2.rectangle(image,(xmin,ymin),
                                                      (xmin + w , ymin + h), (255,255,255),3)#recorte de la imagen
                            height_two = new_image.shape[0]#IMAGE HEIGHT
                            width_two = new_image.shape[1]#IMAGE WIDTH
                            image_processing = face_mesh.process(new_image)
                            # white = np.ones((height_two,width_two,3),dtype =np.uint8)
                            # white = white*255
                            label = label + 1
                            canvas = cv2.resize(rectangle,(640,480),interpolation=cv2.INTER_CUBIC)                    
                            # cv2.imshow('original', canvas)
                            # cv2.imwrite(frame + str(label) + 'o.jpg',canvas)
                            if image_processing.multi_face_landmarks:
                                for face_landmarks in image_processing.multi_face_landmarks:
                                    instanteFinal = datetime.now()
                                    tiempo = instanteFinal - instanteInicial
                                    segundos = tiempo.seconds
                                    # mp_drawing.draw_landmarks(white, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,conf_drawing,conf_drawing)
                                    #HORIZONTAL_LIP
                                    x1 = int(face_landmarks.landmark[61].x * width_two)
                                    y1 = int(face_landmarks.landmark[61].y *  height_two)
                                    x2 = int(face_landmarks.landmark[291].x * width_two)
                                    y2 = int(face_landmarks.landmark[291].y * height_two)
                                    #VERTICAL_LIP
                                    x3 = int(face_landmarks.landmark[13].x * width_two)
                                    y3 = int(face_landmarks.landmark[13].y *  height_two)
                                    x4 = int(face_landmarks.landmark[14].x * width_two)
                                    y4 = int(face_landmarks.landmark[14].y * height_two)
                                    #LEFT_EYE
                                    x5 = int(face_landmarks.landmark[374].x * width_two)
                                    y5 = int(face_landmarks.landmark[374].y *  height_two)
                                    x6 = int(face_landmarks.landmark[386].x * width_two)
                                    y6 = int(face_landmarks.landmark[386].y * height_two)
                                    #RIGHT_EYE
                                    x7 = int(face_landmarks.landmark[159].x * width_two)
                                    y7 = int(face_landmarks.landmark[159].y *  height_two)
                                    x8 = int(face_landmarks.landmark[145].x * width_two)
                                    y8 = int(face_landmarks.landmark[145].y * height_two)
                                    #RIGHT_BROW
                                    x9 = int(face_landmarks.landmark[66].x * width_two)
                                    y9 = int(face_landmarks.landmark[66].y *  height_two)
                                    x10 = int(face_landmarks.landmark[153].x * width_two)
                                    y10 = int(face_landmarks.landmark[153].y * height_two)
                                    #LEFT_BROW
                                    x11 = int(face_landmarks.landmark[380].x * width_two)
                                    y11 = int(face_landmarks.landmark[380].y *  height_two)
                                    x12 = int(face_landmarks.landmark[296].x * width_two)
                                    y12 = int(face_landmarks.landmark[296].y * height_two)
                                    #CHIN
                                    x13 = int(face_landmarks.landmark[2].x * width_two)
                                    y13 = int(face_landmarks.landmark[2].y *  height_two)
                                    x14 = int(face_landmarks.landmark[152].x * width_two)
                                    y14 = int(face_landmarks.landmark[152].y * height_two)
                                    #BETWEEN_EYEBROWS_1
                                    x15 = int(face_landmarks.landmark[55].x * width_two)
                                    y15 = int(face_landmarks.landmark[55].y *  height_two)
                                    x16 = int(face_landmarks.landmark[285].x * width_two)
                                    y16 = int(face_landmarks.landmark[285].y * height_two)
                                    #BETWEEN_EYEBROWS_2
                                    x17 = int(face_landmarks.landmark[107].x * width_two)
                                    y17 = int(face_landmarks.landmark[107].y *  height_two)
                                    x18 = int(face_landmarks.landmark[336].x * width_two)
                                    y18 = int(face_landmarks.landmark[336].y * height_two)
                                    #minima left wing and procerus
                                    x19 = int(face_landmarks.landmark[9].x * width_two)
                                    y19 = int(face_landmarks.landmark[9].y *  height_two)
                                    x20 = int(face_landmarks.landmark[437].x * width_two)
                                    y20 = int(face_landmarks.landmark[437].y * height_two)                            
                                    #maxima right wing and procerus
                                    x21 = int(face_landmarks.landmark[217].x * width_two)
                                    y21 = int(face_landmarks.landmark[217].y *  height_two)
                                    #minima right wing and procerus
                                    x22 = int(face_landmarks.landmark[188].x * width_two)
                                    y22 = int(face_landmarks.landmark[188].y * height_two)                              
                                    #maxima left wing and procerus
                                    x23 = int(face_landmarks.landmark[412].x * width_two)
                                    y23 = int(face_landmarks.landmark[412].y * height_two)                            
                                    
                                    
                                    horizontal_lip.append(int(distance.euclidean((x1, y1), (x2, y2))))
                                    vertical_lip.append(int(distance.euclidean((x3, y3), (x4, y4))))
                                    right_eye.append(int(distance.euclidean((x5, y5), (x6, y6))))
                                    left_eye.append(int(distance.euclidean((x7, y7), (x8, y8))))                           
                                    right_brow.append(int(distance.euclidean((x9, y9), (x10, y10))))
                                    left_brow.append(int(distance.euclidean((x11, y11), (x12, y12))))
                                    chin.append(int(distance.euclidean((x13, y13), (x14, y14))))
                                    between_eyebrows_1.append(int(distance.euclidean((x15, y15), (x16, y16))))
                                    between_eyebrows_2.append(int(distance.euclidean((x17, y17), (x18, y18))))
                                    bridges_1.append(int(distance.euclidean((x19, y19), (x20, y20))))
                                    bridges_2.append(int(distance.euclidean((x19, y19), (x21, y21))))
                                    bridges_3.append(int(distance.euclidean((x19, y19), (x22, y22))))
                                    bridges_4.append(int(distance.euclidean((x19, y19), (x23, y23))))
                                    clase.append(address1)
                                    
                                    contador.append(label)
                                    frames.append(str(int(label/(segundos+0.00001))))
                                    if (label % 10) == 0:
                                                points.append(label)
                                                
                                    # cv2.circle(new_image,(x1, y1),2,(255, 0, 0), t)
                                    # cv2.circle(new_image, (x2, y2),2,(255, 0, 0), t)
                                    # cv2.line(new_image, (x1,y1),(x2,y2),(255,255,255), t)
                                    
                                    # cv2.circle(new_image, (x3, y3), 2, (255, 0, 0), t)
                                    # cv2.circle(new_image, (x4, y4), 2, (255, 0, 0), t)
                                    # cv2.line(new_image, (x3,y3),(x4,y4),(255,255,255), t)
                                    
                                    # cv2.circle(new_image, (x5, y5), 2, (255, 0, 0), t)
                                    # cv2.circle(new_image, (x6, y6), 2, (255, 0, 0), t)
                                    # cv2.line(new_image, (x5,y5),(x6,y6),(255,255,255), t)
                                    
                                    # cv2.circle(new_image, (x7, y7), 2, (255, 0, 0), t)
                                    # cv2.circle(new_image, (x8, y8), 2, (255, 0, 0), t)
                                    # cv2.line(new_image, (x7,y7),(x8,y8),(255,255,255), t)
                                    
                                    # cv2.circle(new_image,(x9, y9),2,(255, 0, 0), t)
                                    # cv2.circle(new_image, (x10, y10),2,(255, 0, 0), t)
                                    # cv2.line(new_image, (x9,y9),(x10,y10),(255,255,255), t)
                                    
                                    # cv2.circle(new_image, (x11, y11), 2, (255, 0, 0), t)
                                    # cv2.circle(new_image, (x12, y12), 2, (255, 0, 0), t)
                                    # cv2.line(new_image, (x11,y11),(x12,y12),(255,255,255), t)
                                    
                                    # cv2.circle(new_image, (x13, y13), 2, (255, 0, 0), t)
                                    # cv2.circle(new_image, (x14, y14), 2, (255, 0, 0), t)
                                    # cv2.line(new_image, (x13,y13),(x14,y14),(255,255,255), t)
                                    
                                    # cv2.circle(new_image, (x15, y15), 2, (255, 0, 0), t)
                                    # cv2.circle(new_image, (x16, y16), 2, (255, 0, 0), t)
                                    # cv2.line(new_image, (x15,y15),(x16,y16),(255,255,255), t)

                                    # cv2.circle(new_image,(x17, y17),2,(255, 0, 0), t)
                                    # cv2.circle(new_image, (x18, y18),2,(255, 0, 0), t)
                                    # cv2.line(new_image, (x17,y17),(x18,y18),(255,255,255), t)
                                    
                                    # cv2.circle(new_image, (x19, y19), 2, (255, 0, 0), t)
                                    # cv2.circle(new_image, (x20, y20), 2, (255, 0, 0), t)
                                    # cv2.line(new_image, (x19,y19),(x20,y20),(255,255,255), t)
                                    
                                    # cv2.circle(new_image, (x21, y21), 2, (255, 0, 0), t)
                                    # cv2.line(new_image, (x19,y19),(x21,y21),(255,255,255), t)
                                    
                                    # cv2.circle(new_image, (x22, y22), 2, (255, 0, 0), t)
                                    # cv2.line(new_image, (x19,y19),(x22,y22),(255,255,255), t)
                                    
                                    # cv2.circle(new_image, (x23, y23), 2, (255, 0, 0), t)
                                    # cv2.line(new_image, (x19,y19),(x23,y23),(255,255,255), t)
                                    
                                    
                                    
                                    # cv2.circle(white,(x1, y1),2,(255, 0, 0), t)
                                    # cv2.circle(white, (x2, y2),2,(255, 0, 0), t)
                                    # cv2.line(white, (x1,y1),(x2,y2),(0,0,255), t)
                                    
                                    # cv2.circle(white, (x3, y3), 2, (255, 0, 0), t)
                                    # cv2.circle(white, (x4, y4), 2, (255, 0, 0), t)
                                    # cv2.line(white, (x3,y3),(x4,y4),(0,0,255), t)
                                    
                                    # cv2.circle(white, (x5, y5), 2, (255, 0, 0), t)
                                    # cv2.circle(white, (x6, y6), 2, (255, 0, 0), t)
                                    # cv2.line(white, (x5,y5),(x6,y6),(0,0,255), t)
                                    
                                    # cv2.circle(white, (x7, y7), 2, (255, 0, 0), t)
                                    # cv2.circle(white, (x8, y8), 2, (255, 0, 0), t)
                                    # cv2.line(white, (x7,y7),(x8,y8),(0,0,255), t)
                                    
                                    # cv2.circle(white,(x9, y9),2,(255, 0, 0), t)
                                    # cv2.circle(white, (x10, y10),2,(255, 0, 0), t)
                                    # cv2.line(white, (x9,y9),(x10,y10),(0,0,255), t)
                                    
                                    # cv2.circle(white, (x11, y11), 2, (255, 0, 0), t)
                                    # cv2.circle(white, (x12, y12), 2, (255, 0, 0), t)
                                    # cv2.line(white, (x11,y11),(x12,y12),(0,0,255), t)
                                    
                                    # cv2.circle(white, (x13, y13), 2, (255, 0, 0), t)
                                    # cv2.circle(white, (x14, y14), 2, (255, 0, 0), t)
                                    # cv2.line(white, (x13,y13),(x14,y14),(0,0,255), t)
                                    
                                    # cv2.circle(white, (x15, y15), 2, (255, 0, 0), t)
                                    # cv2.circle(white, (x16, y16), 2, (255, 0, 0), t)
                                    # cv2.line(white, (x15,y15),(x16,y16),(0,0,255), t)

                                    # cv2.circle(white,(x17, y17),2,(255, 0, 0), t)
                                    # cv2.circle(white, (x18, y18),2,(255, 0, 0), t)
                                    # cv2.line(white, (x17,y17),(x18,y18),(0,0,255), t)
                                    
                                    # cv2.circle(white, (x19, y19), 2, (255, 0, 0), t)
                                    # cv2.circle(white, (x20, y20), 2, (255, 0, 0), t)
                                    # cv2.line(white, (x19,y19),(x20,y20),(0,0,255), t)
                                    
                                    # cv2.circle(white, (x21, y21), 2, (255, 0, 0), t)
                                    # cv2.line(white, (x19,y19),(x21,y21),(0,0,255), t)
                                    
                                    # cv2.circle(white, (x22, y22), 2, (255, 0, 0), t)
                                    # cv2.line(white, (x19,y19),(x22,y22),(0,0,255), t)
                                    
                                    # cv2.circle(white, (x23, y23), 2, (255, 0, 0), t)
                                    # cv2.line(white, (x19,y19),(x23,y23),(0,0,255), t)

                                    # canva = cv2.resize(new_image,(640,480),interpolation=cv2.INTER_CUBIC)
                                    # cv2.imwrite(frame + str(label) + 'r.jpg',canva)
                                    # cv2.imshow('rostro_detectado', canva)
                                    # canvap = cv2.resize(white,(640,480),interpolation=cv2.INTER_CUBIC)
                                    # cv2.imwrite(frame + str(label) + 'rp.jpg',canvap)
                                    # cv2.imshow('rostro_detectado', canvap)
                                    # video_output.write(canvap)
                                    # video_output3.write(canva)

                    # video_output2.write(canvas)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
                else:
                    break
             
            horizontal_lip = horizontal_lip/np.linalg.norm(horizontal_lip)
            vertical_lip = vertical_lip/np.linalg.norm(vertical_lip)
            left_eye = left_eye/np.linalg.norm(left_eye)
            right_eye = right_eye/np.linalg.norm(right_eye)
            right_brow = right_brow/np.linalg.norm(right_brow)
            left_brow = left_brow/np.linalg.norm(left_brow)
            chin = chin/np.linalg.norm(chin)
            between_eyebrows_1 = between_eyebrows_1/np.linalg.norm(between_eyebrows_1)
            between_eyebrows_2 = between_eyebrows_2/np.linalg.norm(between_eyebrows_2)
            bridges_1 = bridges_1/np.linalg.norm(bridges_1)
            bridges_2 = bridges_2/np.linalg.norm(bridges_2)
            bridges_3 = bridges_3/np.linalg.norm(bridges_3)
            bridges_4 = bridges_4/np.linalg.norm(bridges_4)
            
            fig, ax = plt.subplots()
            ax.plot(contador,horizontal_lip,label = '1_horizontal_lip', color = 'r')
            ax.plot(contador,vertical_lip,label = '2_vertical_lip',color = 'b')
            ax.plot(contador,right_eye,label = '4_right_eye',color = 'g')
            ax.plot(contador,left_eye,label = '3_left_eye',color = 'k')
            ax.plot(contador,right_brow,label = '5_right_brow', color = 'y')
            ax.plot(contador,left_brow,label = '6_left_brow',color = 'c')
            ax.plot(contador,chin,label = '7_chin_and_nose',color = 'm')
            lg = ax.legend(fontsize = 6.5,bbox_to_anchor=(1.1,1.07),ncol = 4,loc ='right')
            ax.set_xticks(points)
            ax.set_xticklabels(points)
            plt.tight_layout()
            plt.ylabel('DISTANCES',fontsize = 8)
            plt.xlabel('FRAMES',fontsize = 8)
            plt.savefig(str(grafica) +'1.jpg',dpi = 400)
            plt.close()
            
            fig, ax = plt.subplots()
            ax.plot(contador,between_eyebrows_1,label = '8_between_eyebrows_lower',color = 'r')
            ax.plot(contador,between_eyebrows_2,label = '9_between_eyebrows_superior',color = 'b')
            ax.plot(contador,bridges_1,label = '10_minima_left_wing_and_procerus',color = 'g')
            ax.plot(contador,bridges_2,label = '12_maxima_right_wing_and_procerus',color = 'k')
            ax.plot(contador,bridges_3,label = '11_minima_right_wing_and_procerus',color = 'c')
            ax.plot(contador,bridges_4,label = '13_maxima_left_wing_and_procerus',color = 'm')
            lg = ax.legend(fontsize = 6.5,bbox_to_anchor=(1.1,1.07),ncol = 3,loc ='right')
            ax.set_xticks(points)
            ax.set_xticklabels(points)
            plt.tight_layout()
            plt.ylabel('DISTANCES',fontsize = 8)
            plt.xlabel('FRAMES',fontsize = 8)
            plt.savefig(str(grafica) +'2.jpg',dpi = 400)
            plt.close()
            
            points = pd.DataFrame({'frames':frames,'horizontal_lip':horizontal_lip,
                                    'vertical_lip':vertical_lip,'right_eye':right_eye,'left_eye':left_eye,'right_brow':right_brow,
                                    'left_brow':left_brow,'chin':chin,'between_eyebrows_1':between_eyebrows_1,'between_eyebrows_2':between_eyebrows_2,
                                    'bridge_1':bridges_1,'bridge_2':bridges_2,'bridge_3':bridges_3,'bridge_4':bridges_4,'clase':clase})
            points = points[['frames','horizontal_lip',
                              'vertical_lip','right_eye','left_eye','right_brow',
                              'left_brow','chin','between_eyebrows_1','between_eyebrows_2',
                              'bridge_1','bridge_2','bridge_3','bridge_4','clase']]
            writer = ExcelWriter(datos_excel + '.xlsx')
            points.to_excel(writer, 'MEDIDAS' , index=False)
            writer.save()
            writer.close()
            capture.release()
cv2.destroyAllWindows()
# video_output.release()
# video_output2.release()
# video_output3.release()
instanteFinal2 = datetime.now()
tiempo2 = instanteFinal2 - instanteInicial
print(tiempo2)


