# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 00:10:50 2022

@author: MARCOS
"""
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,CSVLogger
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LeakyReLU
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np
import keras
import sys
import os
now = datetime.now()
lear = [0.001,0.0001,0.00001]
epoc = [200,400,800]
neur = [32,64,128]
cap = 3

drive = "C:/Users/MCIM/Documents/MARCOS/DATOS"

'''cargamos el datapath'''
def load_data(data_path):
    
    print(data_path)
    data_file = f'{data_path}dat.npy'
    label_file = f'{data_path}lab.npy'
    
    if os.path.exists(data_file) == True:
        X = np.load(data_file)
        y = np.load(label_file)
        
    else:
        print('"file not exist"')
        sys.exit()    
   
    return X, y

def myprint(s):
    with open(sumary,'w') as f:
        model.summary(print_fn=lambda x: f.write(f'{x}\n'))

'''preparamos los datos para el train y el test'''
def prepare_datasets(val_size):
    
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_size, random_state= 0, stratify = y)
    return X_train, X_test, y_train, y_test

def plot_history(history):
    fig, axs = plt.subplots(2, figsize=(16,10))

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval", fontsize=35, loc='right')

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval",fontsize=35,loc='right')
    
    
    plt.tight_layout()
    plt.savefig(resultados+"lost_acuracy.jpg")
    plt.clf()

    '''creamos la matriz de confusion'''
def plot_matrix(X_test,y_test):
    model_probs = model.predict(X_test)
    y_pred = np.argmax(model_probs,axis=1)
    y_test2 = np.argmax(y_test,axis=1)
    conf_mat = confusion_matrix(y_test2,y_pred)
    conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[ :,np.newaxis]
    df_cm = pd.DataFrame(conf_mat, index=['Angry','Happy','Surprised','sad'], columns=['Angry','Happy','Surprised','sad'])
    plt.figure(figsize = (10,8))
    plt.title('Confusion Matrix', fontsize=35, loc='right')
    res=sns.heatmap(df_cm, annot=True, fmt=".1%", annot_kws={'size':31}, cmap='coolwarm',cbar=False)
    res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 17)
    res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 17)
    res.set_xlabel('Actual label',fontsize = 25)
    res.set_ylabel('Predicted label',fontsize = 25)
    plt.savefig(resultados +"matrix_confusion2.jpg")
    plt.clf()

for i in lear:
    for j in epoc:
        for k in neur:
            for l in range(cap):
                
                '''direccion de la base de datos'''
                epocas = j
                learning_rate = i
                num = k
                direccion = f'blstm_{num}_epocas_{epocas}_lr_{learning_rate}_numcap_{l}/'
                DATA_PATH = f'{drive}/CK/'
                resultados =f'{drive}/models_final_cohn_kanade/{direccion}'
                sumary = f'{resultados}model_summary.txt'
                model_file = f'{resultados}blstm_model.h5'
                caracteristicas = f'Learning cup: blstm\n Number of epochs: {epocas}\n Learning rate: {learning_rate}\n Number of hidden layers: {l}\n Number of neurons in layer: {k}'
    
    
                if not os.path.exists(resultados):#THE ADDRESS WHERE IT WILL BE SAVED IS CREATED
                    os.makedirs(resultados)
    
                '''funcion para reducir la taza de aprendizaje'''
                def step_decay(epoc):
                    x=0.0001
                    decay_rate =0.9
                    if epoc >=100: x=x*decay_rate
                    if epoc >=200: x=x*(decay_rate*decay_rate)
                    if epoc >=300: x=x*(decay_rate*decay_rate*decay_rate)
                    return x
    
                '''construimos el modelo'''
                def build_model_0(input_shape):
                    
                    '''creamos el contenedor de la red LSTM'''
                    model = Sequential(name='model_BLSTM_step_decay')
                    
                    '''numero de neuronas y tamaño de la capa de entrada'''
                    model.add(Bidirectional(LSTM(num, input_shape=input_shape, return_sequences=True),name='input_layer'))
                    model.add(Bidirectional(LSTM(num),name='layer_1'))
                    model.add(Dropout(0.2,name='dropout_1'))
                   
                    '''capa de salida'''
                    model.add(Dense(4, activation='softmax',name='output_layer'))
                    return model
                
                def build_model_1(input_shape):
                    
                    '''creamos el contenedor de la red LSTM'''
                    model = Sequential(name='model_BLSTM_step_decay')
                    
                    '''numero de neuronas y tamaño de la capa de entrada'''
                    model.add(Bidirectional(LSTM(num, input_shape=input_shape, return_sequences=True),name='input_layer'))
                    model.add(Bidirectional(LSTM(num),name='layer_1'))
                    model.add(Dropout(0.2,name='dropout_1'))
                    #dense layer
                    model.add(Dense(num,name='dense_2'))
                    model.add(LeakyReLU(alpha=0.05,name='LeakyReLU_2'))
                    model.add(Dropout(0.2,name='dropout_2'))
                   
                    '''capa de salida'''
                    model.add(Dense(4, activation='softmax',name='output_layer'))
                    return model
                
                def build_model_2(input_shape):
                    
                    '''creamos el contenedor de la red LSTM'''
                    model = Sequential(name='model_BLSTM_step_decay')
                    
                    '''numero de neuronas y tamaño de la capa de entrada'''
                    model.add(Bidirectional(LSTM(num, input_shape=input_shape, return_sequences=True),name='input_layer'))
                    model.add(Bidirectional(LSTM(num),name='layer_1'))
                    model.add(Dropout(0.2,name='dropout_1'))
                    #dense layer
                    model.add(Dense(num,name='dense_2'))
                    model.add(LeakyReLU(alpha=0.05,name='LeakyReLU_2'))
                    model.add(Dropout(0.2,name='dropout_2'))
                    #dense layer
                    model.add(Dense(num,name='dense_3'))
                    model.add(LeakyReLU(alpha=0.05,name='LeakyReLU_3'))
                    model.add(Dropout(0.2,name='dropout_3'))
                   
                    '''capa de salida'''
                    model.add(Dense(4, activation='softmax',name='output_layer'))
                    return model

                '''separamos los datos para el entrenamiento y test'''
                X_train, X_test, y_train, y_test = prepare_datasets(0.2)
    
                '''construimos el modelo con los datos para el train'''
                input_shape = (X_train.shape[1], X_train.shape[2])
                
                if l == 0:
                    model = build_model_0(input_shape)
                elif l == 1:
                    model = build_model_1(input_shape)
                elif l == 2:
                    model = build_model_2(input_shape)

                '''declaramos el optimizador y compilamos el modelo'''
                optimiser = Adam(learning_rate)
                model.compile(optimizer=optimiser,
                                  loss='categorical_crossentropy',
                                  metrics=['accuracy'])
    
                '''historial de metricas'''
                lr_decay = LearningRateScheduler(step_decay)
                csv_logger = CSVLogger(resultados + "datos.csv" , append=False, separator=',')
                model_checkpoint = ModelCheckpoint(filepath=model_file, monitor='val_accuracy',verbose=1,
                                                    save_best_only=True, save_weights_only=False,mode='auto',
                                                    period=1)
                history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                                    batch_size=64, epochs=epocas,callbacks=[csv_logger,model_checkpoint, lr_decay])
    
                '''visualizamos el modelo'''
                model.summary(print_fn=myprint)
    
                '''graficamos los resultados accuracy/error'''
                plot_history(history)
    
                '''creamos la matriz de confusion'''
                plot_matrix(X_test,y_test)
    
                '''evaluamos el modelo con el set de test'''
                test_loss0, test_acc0 = model.evaluate(X_test, y_test, verbose=2)
    
                '''usamos el modelo con el set de test'''
                best_model = load_model(model_file)
                test_loss1, test_acc1 = best_model.evaluate(X_test, y_test, verbose=2)
                
                with open(f'{resultados} blstm_{test_acc0}.txt',"w") as file:
                    file.write(f'\nTest accuracy eval: {test_acc0}')
                    file.write(f'\nTest lossy eval: {test_loss0}')
                    file.write(f'\nTest accuracy model: {test_acc1}')
                    file.write(f'\nTest lossy model: {test_loss1}')
                    file.write(f'\nCARACTERISTICAS\n{caracteristicas}')
                    file.close()

for i in lear:
    for j in epoc:
        for k in neur:
            for l in range(cap):
                
                '''direccion de la base de datos'''
                epocas = j
                learning_rate = i
                num = k

                direccion = f'lr_time_based_decay_{num}_epocas_{epocas}_lr_{learning_rate}_numcap_{l}/'
                DATA_PATH = f'{drive}/CK/'
                resultados =f'{drive}/models_final_cohn_kanade/{direccion}'
                sumary = f'{resultados}model_summary.txt'
                model_file = f'{resultados}blstm_model.h5'
                caracteristicas = f'Learning cup: lr time based decay\n Number of epochs: {epocas}\n Learning rate: {learning_rate}\n Number of hidden layers: {l}\n Number of neurons in layer: {k}'
    
                decay = learning_rate / epocas
                
                if not os.path.exists(resultados):#THE ADDRESS WHERE IT WILL BE SAVED IS CREATED
                    os.makedirs(resultados)
                '''funcion para reducir la taza de aprendizaje'''
                def lr_time_based_decay(epoc, lr):
                    return lr * 1 / (1 + decay * epoc)
                           
                '''construccion del modelo'''
                def build_model_0(input_shape):
                    
                    '''creamos el contenedor de la red LSTM'''
                    model = Sequential(name='model_LSTM_lr_time_based_decay')
                    
                    '''numero de neuronas y tamaño de la capa de entrada'''
                    model.add(LSTM(num, input_shape=input_shape, return_sequences=True,name='input_layer'))
                    model.add(LSTM(num,name='layer_1'))
                    model.add(Dropout(0.2,name='dropout_1'))
                   
                    '''capa de salida'''
                    model.add(Dense(4, activation='softmax',name='layer_output'))
                    return model
                
                def build_model_1(input_shape):
                    
                    '''creamos el contenedor de la red LSTM'''
                    model = Sequential(name='model_LSTM_lr_time_based_decay')
                    
                    '''numero de neuronas y tamaño de la capa de entrada'''
                    model.add(LSTM(num, input_shape=input_shape, return_sequences=True,name='input_layer'))
                    model.add(LSTM(num,name='layer_1'))
                    model.add(Dropout(0.2,name='dropout_1'))
                    #dense layer
                    model.add(Dense(num,name='dense_2'))
                    model.add(LeakyReLU(alpha=0.05,name='LeakyReLU_2'))
                    model.add(Dropout(0.2,name='dropout_2'))
                   
                    '''capa de salida'''
                    model.add(Dense(4, activation='softmax',name='layer_output'))
                    return model
                
                def build_model_2(input_shape):
                    
                    '''creamos el contenedor de la red LSTM'''
                    model = Sequential(name='model_LSTM_lr_time_based_decay')
                    
                    '''numero de neuronas y tamaño de la capa de entrada'''
                    model.add(LSTM(num, input_shape=input_shape, return_sequences=True,name='input_layer'))
                    model.add(LSTM(num,name='layer_1'))
                    model.add(Dropout(0.2,name='dropout_1'))
                    #dense layer
                    model.add(Dense(num,name='dense_2'))
                    model.add(LeakyReLU(alpha=0.05,name='LeakyReLU_2'))
                    model.add(Dropout(0.2,name='dropout_2'))
                    #dense layer
                    model.add(Dense(num,name='dense_3'))
                    model.add(LeakyReLU(alpha=0.05,name='LeakyReLU_3'))
                    model.add(Dropout(0.2,name='dropout_3'))
                    '''capa de salida'''
                    model.add(Dense(4, activation='softmax',name='layer_output'))
                    return model
                
                '''separamos los datos para el entrenamiento y test'''
                X_train, X_test, y_train, y_test = prepare_datasets(0.2)
                '''construimos el modelo con los datos para el train'''
                input_shape = (X_train.shape[1], X_train.shape[2]) 
                if l == 0:
                    model = build_model_0(input_shape)
                elif l == 1:
                    model = build_model_1(input_shape)
                elif l == 2:
                    model = build_model_2(input_shape)
                               
                '''declaramos el optimizador y compilamos el modelo'''
                optimiser = keras.optimizers.Adam(learning_rate)
                model.compile(optimizer=optimiser,
                                   loss='categorical_crossentropy',
                                   metrics=['accuracy'])
                
                '''historial de metricas'''
                callback_lr_time_based_decay = tf.keras.callbacks.LearningRateScheduler(lr_time_based_decay)
                csv_logger = CSVLogger(resultados + "datos.csv" , append=False, separator=',')
                model_checkpoint = ModelCheckpoint(filepath=model_file, monitor='val_accuracy',verbose=1,
                                                    save_best_only=True, save_weights_only=False,mode='auto',
                                                    period=1)
                history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                                    batch_size=64, epochs=epocas,callbacks=[csv_logger,callback_lr_time_based_decay,model_checkpoint])
               
               '''visualizamos el modelo'''
                model.summary(print_fn=myprint)
                
                '''graficamos los resultados accuracy/error'''
                plot_history(history)
    
                '''creamos la matriz de confusion'''
                plot_matrix(X_test,y_test)
                
                '''evaluamos el modelo con el set de test'''
                test_loss0, test_acc0 = model.evaluate(X_test, y_test, verbose=2)
    
                '''usamos el modelo con el set de test'''
                best_model = load_model(model_file)
                test_loss1, test_acc1 = best_model.evaluate(X_test, y_test, verbose=2)
                
                with open(f'{resultados} lr_time_based_decay_{test_acc0}.txt',"w") as file:
                    file.write(f'\nTest accuracy eval: {test_acc0}')
                    file.write(f'\nTest lossy eval: {test_loss0}')
                    file.write(f'\nTest accuracy model: {test_acc1}')
                    file.write(f'\nTest lossy model: {test_loss1}')
                    file.write(f'\nCARACTERISTICAS\n{caracteristicas}')
                    file.close()

for i in lear:
    for j in epoc:
        for k in neur:
            for l in range(cap):
                
                '''direccion de la base de datos'''
                epocas = j
                learning_rate = i
                num = k
                direccion = f'scheduler_{num}_epocas_{epocas}_lr_{learning_rate}_numcap_{l}/'
                DATA_PATH = f'{drive}/CK/'
                resultados =f'{drive}/models_final_cohn_kanade/{direccion}'
                sumary = f'{resultados}model_summary.txt'
                model_file = f'{resultados}blstm_model.h5'
                caracteristicas = f'Learning cup: scheduler\n Number of epochs: {epocas}\n Learning rate: {learning_rate}\n Number of hidden layers: {l}\n Number of neurons in layer: {k}'
    
                
                if not os.path.exists(resultados):#THE ADDRESS WHERE IT WILL BE SAVED IS CREATED
                    os.makedirs(resultados)
                '''funcion para reducir la taza de aprendizaje'''
                def scheduler(epoc, lr):
                    if epoc <= 10:
                        return lr
                    else:
                        return lr * tf.math.exp(-0.01)
    
                def build_model_0(input_shape):
                    
                    '''creamos el contenedor de la red LSTM'''
                    model = Sequential(name='model_LSTM_scheduler')
                    
                    '''numero de neuronas y tamaño de la capa de entrada'''
                    model.add(LSTM(num, input_shape=input_shape, return_sequences=True,name='input_layer'))
                    model.add(LSTM(num,name='layer_1'))
                    model.add(Dropout(0.2,name='dropout_1'))
                   
                    '''capa de salida'''
                    model.add(Dense(4, activation='softmax',name='layer_output'))
                    return model
                
                def build_model_1(input_shape):
                    
                    '''creamos el contenedor de la red LSTM'''
                    model = Sequential(name='model_LSTM_scheduler')
                    
                    '''numero de neuronas y tamaño de la capa'''
                    model.add(LSTM(num, input_shape=input_shape, return_sequences=True,name='input_layer'))
                    model.add(LSTM(num,name='layer_1'))
                    model.add(Dropout(0.2,name='dropout_1'))
                    #dense layer
                    model.add(Dense(num,name='dense_2'))
                    model.add(LeakyReLU(alpha=0.05,name='LeakyReLU_2'))
                    model.add(Dropout(0.2,name='dropout_2'))
                   
                    '''capa de salida'''
                    model.add(Dense(4, activation='softmax',name='layer_output'))
                    return model
                
                def build_model_2(input_shape):
                    
                    '''creamos el contenedor de la red LSTM'''
                    model = Sequential(name='model_LSTM_scheduler')
                    
                    '''numero de neuronas y tamaño de la capa'''
                    model.add(LSTM(num, input_shape=input_shape, return_sequences=True,name='input_layer'))
                    model.add(LSTM(num,name='layer_1'))
                    model.add(Dropout(0.2,name='dropout_1'))
                    #dense layer
                    model.add(Dense(num,name='dense_2'))
                    model.add(LeakyReLU(alpha=0.05,name='LeakyReLU_2'))
                    model.add(Dropout(0.2,name='dropout_2'))
                    #dense layer
                    model.add(Dense(num,name='dense_3'))
                    model.add(LeakyReLU(alpha=0.05,name='LeakyReLU_3'))
                    model.add(Dropout(0.2,name='dropout_3'))
                    '''capa de salida'''
                    model.add(Dense(4, activation='softmax',name='layer_output'))
                    return model
                
                '''separamos los datos para el entrenamiento y test'''
                X_train, X_test, y_train, y_test = prepare_datasets(0.2)
                
                '''construimos el modelo con los datos para el train'''
                input_shape = (X_train.shape[1], X_train.shape[2]) 
                if l == 0:
                    model = build_model_0(input_shape)
                elif l == 1:
                    model = build_model_1(input_shape)
                elif l == 2:
                    model = build_model_2(input_shape)

                '''declaramos el optimizador y compilamos el modelo'''
                optimiser = keras.optimizers.Adam(learning_rate)
                model.compile(optimizer=optimiser,
                                   loss='categorical_crossentropy',
                                   metrics=['accuracy'])
                
                '''historial de metricas'''
                callback_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
                csv_logger = CSVLogger(resultados + "datos.csv" , append=False, separator=',')
                model_checkpoint = ModelCheckpoint(filepath=model_file, monitor='val_accuracy',verbose=1,
                                                    save_best_only=True, save_weights_only=False,mode='auto',
                                                    period=1)
                history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                                    batch_size=64, epochs=epocas,callbacks=[csv_logger,callback_scheduler,model_checkpoint])
                
                '''visualizamos el modelo'''
                model.summary(print_fn=myprint)
                
                '''graficamos los resultados accuracy/error'''
                plot_history(history)
    
                '''creamos la matriz de confusion'''
                plot_matrix(X_test,y_test)
                      
                '''evaluamos el modelo con el set de test'''
                test_loss0, test_acc0 = model.evaluate(X_test, y_test, verbose=2)
    
                '''usamos el modelo con el set de test'''
                best_model = load_model(model_file)
                test_loss1, test_acc1 = best_model.evaluate(X_test, y_test, verbose=2)
                
                with open(f'{resultados} scheduler_{test_acc0}.txt',"w") as file:
                    file.write(f'\nTest accuracy eval: {test_acc0}')
                    file.write(f'\nTest lossy eval: {test_loss0}')
                    file.write(f'\nTest accuracy model: {test_acc1}')
                    file.write(f'\nTest lossy model: {test_loss1}')
                    file.write(f'\nCARACTERISTICAS\n{caracteristicas}')
                    file.close()
                    
new = datetime.now()
with open ( f'{drive}:/models_final/tiempo_de_ejecucion.txt','w') as f:
    file.write(f'\ninicio de la ejecucion schdlr: {now.day}-{now.month}-{now.year}  {now.hour}:{now.minute}:{now.second}')
    file.write(f'\nfin de la ejecucion schdlr: {new.year}-{new.month}-{new.day}  {new.hour}:{new.minute}:{new.second}')
    file.close()
