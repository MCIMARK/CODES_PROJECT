# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 15:41:05 2023

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
import keras_tuner as kt
import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np
import keras
import sys
import os


now = datetime.now()
print(f'\nINICIO DE EJECUCION LTBDM: {now.day}-{now.month}-{now.year}  {now.hour}:{now.minute}:{now.second}')
epochs = 800
learning_rate = 0.00001
decay = learning_rate / epochs

drive = 'C:/Users/MCIM/Documents/MARCOS/para_entrenamiento'
resultados =f'{drive}/modelo_full_ltbdm/'
datos = f'{drive}/full_database_project/'
sumary = f'{resultados}ltbdm_model_summary.txt'
model_file = f'{resultados}ltbdm_modelo.h5'

'''creamos la carpeta para guardar los resultados'''
if not os.path.exists(resultados):
    os.makedirs(resultados)

'''funcion para imprimir el modelo'''
def myprint(s):
    with open(sumary,'w') as f:
        model.summary(print_fn=lambda x: f.write(f'{x}\n'))

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

'''generamos la imagen del Accuracy y el Error'''
def plot_history(history):
    fig, axs = plt.subplots(2, figsize=(16,10))

    # creamos la imagen del accuracy
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy", fontsize=15)
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval", fontsize=20, loc='right')

    # creamos la imagen del error
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error", fontsize=15)
    axs[1].set_xlabel("Epoch", fontsize=15)
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval",fontsize=20,loc='right')
    
    #guardamos la imagen
    plt.tight_layout()
    plt.savefig(f'{resultados}lost_acuracy.jpg')
    plt.clf()

'''creamos la matriz de confusion'''
def plot_matrix(X_test,y_test):
    model_probs = model.predict(X_test,verbose=False)
    y_pred = np.argmax(model_probs,axis=1)
    y_test2 = np.argmax(y_test,axis=1)
    conf_mat = confusion_matrix(y_pred, y_test2)
    df_cm = pd.DataFrame(conf_mat, index=['Angry','Happy','Surprised', 'Sad'], columns=['Angry','Happy','Surprised', 'Sad'])
    plt.figure(figsize = (10,8))
    plt.title('Confusion Matrix', fontsize=20, loc='right')
    res=sns.heatmap(df_cm, annot=True, fmt=".0f", annot_kws={'size':32}, cmap='coolwarm',cbar=False)
    res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 17)
    res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 17)
    
    #guardamos la imagen
    plt.savefig(f'{resultados}matrix_confusion.jpg')
    plt.clf()
    
'''preparamos los datos para el train y el test'''
def prepare_datasets(val_size):
    
    X, y = load_data(datos)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_size, random_state= 0, stratify = y)
    return X_train, X_test, y_train, y_test

'''separamos los datos para el entrenamiento y test'''
X_train, X_test, y_train, y_test = prepare_datasets(0.2)

'''cargamos los datos para el entrenamiento'''
input_shape = (X_train.shape[1], X_train.shape[2])







'''::::::::::::::::::::MODELO LTBDM:::::::::::::::::::::::::::::'''
m=0
'''construimos el modelo a sintonizar'''
def build_model(hp):
    '''creamos el contenedor de la red LSTM'''
    model = keras.Sequential(name='model_LSTM')
    '''numero de neuronas y tamaño de la capa de entrada'''
    model.add(keras.layers.LSTM(hp.Choice('input_layer',values=[32,64]),return_sequences=True,input_shape=input_shape,name = 'input_layer'))
    '''iteramos en un rango de 1 a 10 capas ocultas'''
    for i in range (hp.Int('hidden_layers',min_value = 1, max_value=3)):
        model.add(keras.layers.LSTM(hp.Choice(f'hidden_layer_{i}',values=[32,64]),return_sequences=True,name=f'hidden_layer_{i}'))
        model.add(LeakyReLU(alpha=0.05,name=f'leakyrelu_{i}'))
        model.add(Dropout(0.2,name=f'dropout_{i}'))
    model.add(keras.layers.LSTM(hp.Choice(f'hidden_layer_{i+1}',values=[32,64]),name=f'hidden_layer_{i+1}'))
    model.add(LeakyReLU(alpha=0.05,name=f'leakyrelu_{i+1}'))
    model.add(Dropout(0.2,name=f'dropout_{i+1}'))
    '''capa de salida'''
    model.add(keras.layers.Dense(4, activation='softmax', name='output_layer_softmax'))
    model.compile(optimizer=Adam(0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    global m
    m+=1
    print(f'modelo: {m}')
    return model

'''instanciamos el sintonizador con los hyperparametros'''
tuner = kt.Hyperband(build_model,#Sintonizador hiperbanda
                      objective='val_accuracy',
                      max_epochs=epochs,
                      factor=3,
                      hyperband_iterations=1,
                      directory=resultados,
                      project_name="trial_ltbdm",
                      overwrite=True)

'''funcion para reducir la taza de aprendizaje'''
def lr_time_based_decay(epochs, lr):
    return lr * 1 / (1 + decay * epochs)

'''procedemos con el ajuste'''
callback_lr_time_based_decay = tf.keras.callbacks.LearningRateScheduler(lr_time_based_decay)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)
tuner.search(X_train, y_train, validation_data=(X_test, y_test),epochs=epochs,callbacks=[stop_early,callback_lr_time_based_decay],use_multiprocessing=True,verbose=False)

'''extraemos los hiperparametros'''
best_hps= tuner.get_best_hyperparameters()[0]
model = tuner.hypermodel.build(best_hps)

'''re-entrenamos el mejor modelo'''
model_checkpoint = ModelCheckpoint(filepath=model_file, monitor='val_accuracy',
                                    save_best_only=True, save_weights_only=False,mode='auto',verbose=False,period=1)
csv_logger = CSVLogger(resultados + "datos.csv" , append=False, separator=',')
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),batch_size=64, epochs=epochs,callbacks=[csv_logger,callback_lr_time_based_decay,model_checkpoint],use_multiprocessing=True,verbose=2)

'''visualizamos el modelo'''
model.summary(print_fn=myprint)

'''visualizamos el modelo'''

'''graficamos los resultados accuracy/error'''
plot_history(history)

'''creamos la matriz de confusion'''
plot_matrix(X_test,y_test)

'''evaluamos el modelo con el set de test'''
test_loss0, test_acc0 = model.evaluate(X_test, y_test, verbose=False)

'''usamos el modelo con el set de test'''
best_model = load_model(model_file)
test_loss1, test_acc1 = best_model.evaluate(X_test, y_test, verbose=False)

'''guardamos los resulatos'''
with open(f'{resultados}ltbdm_{test_acc0}.txt',"w") as file:
    file.write(f'\nTest accuracy eval: {test_acc0}')
    file.write(f'\nTest loss eval: {test_loss0}')
    file.write(f'\nTest accuracy model: {test_acc1}')
    file.write(f'\nTest loss model: {test_loss1}')
    file.write(f'\n Entrenamientos: {m}')
    file.close()

new = datetime.now()
with open(f'{resultados}ltbdm_tiempo.txt',"w") as file:
    file.write(f'\ninicio de la ejecucion ltbdm: {now.day}-{now.month}-{now.year}  {now.hour}:{now.minute}:{now.second}')
    file.write(f'\nfin de la ejecucion ltbdm: {new.year}-{new.month}-{new.day}  {new.hour}:{new.minute}:{new.second}')
    file.close()






''':::::::::::::MODELO STD::::::::::::::'''
now = datetime.now()
print(f'\nINICIO DE EJECUCION STD: {now.day}-{now.month}-{now.year}  {now.hour}:{now.minute}:{now.second}')

resultados =f'{drive}/modelo_full_std/'
datos = f'{drive}/full_database_project/'
sumary = f'{resultados}std_model_summary.txt'
model_file = f'{resultados}std_modelo.h5'

'''creamos la carpeta para guardar los resultados'''
if not os.path.exists(resultados):
    os.makedirs(resultados)

n=0
'''construimos el modelo a sintonizar'''
def build_model(hp):
    '''creamos el contenedor de la red BLSTM'''
    model = keras.Sequential(name='model_BLSTM')
    '''numero de neuronas y tamaño de la capa de entrada'''
    model.add(Bidirectional(LSTM(hp.Choice('input_layer',values=[32,64]),return_sequences=True,input_shape=input_shape),name = 'input_layer'))
    '''iteramos en un rango de 1 a 10 capas ocultas'''
    for i in range (hp.Int('hidden_layers',min_value = 1, max_value=3)):
        model.add(Bidirectional(LSTM(hp.Choice(f'hidden_layer_{i}',values=[32,64]),return_sequences=True),name=f'hidden_layer_{i}'))
        model.add(LeakyReLU(alpha=0.05,name=f'leakyrelu_{i}'))
        model.add(Dropout(0.2,name=f'dropout_{i}'))
    model.add(Bidirectional(LSTM(hp.Choice(f'hidden_layer_{i+1}',values=[32,64])),name=f'hidden_layer_{i+1}'))
    model.add(LeakyReLU(alpha=0.05,name=f'leakyrelu_{i+1}'))
    model.add(Dropout(0.2,name=f'dropout_{i+1}'))
    '''capa de salida'''
    model.add(Dense(4, activation='softmax', name='output_layer_softmax'))
    model.compile(optimizer=Adam(0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    global n
    n+=1
    print(f'modelo: {n}')
    return model

'''instanciamos el sintonizador con los hyperparametros'''
tuner = kt.Hyperband(build_model,#Sintonizador hiperbanda
                      objective='val_accuracy',
                      max_epochs=epochs,
                      factor=3,
                      hyperband_iterations=1,
                      directory=resultados,
                      project_name="trial_ltbdm",
                      overwrite=True)

'''funcion para reducir la taza de aprendizaje'''
def step_decay(epoc):
    x = learning_rate
    decay_rate = 0.9
    if epoc >= 100: x = x*decay_rate
    if epoc >= 200: x = x*(decay_rate*decay_rate)
    if epoc >= 300: x = x*(decay_rate*decay_rate*decay_rate)
    return x

'''procedemos con el ajuste'''
lr_decay = LearningRateScheduler(step_decay)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)
tuner.search(X_train, y_train, validation_data=(X_test, y_test),epochs=epochs, callbacks=[stop_early,lr_decay],use_multiprocessing=True,verbose=False)

'''extraemos los hiperparametros'''
best_hps= tuner.get_best_hyperparameters()[0]
model = tuner.hypermodel.build(best_hps)

'''re-entrenamos el mejor modelo'''
model_checkpoint = ModelCheckpoint(filepath=model_file, monitor='val_accuracy',
                                    save_best_only=True, save_weights_only=False,mode='auto',verbose=False,period=1)
csv_logger = CSVLogger(f'{resultados}datos.csv', append=False, separator=',')
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                                    batch_size=64, epochs=epochs,callbacks=[csv_logger,
                                                                            model_checkpoint, lr_decay],use_multiprocessing=True,verbose=2)
'''visualizamos el modelo'''
model.summary(print_fn=myprint)

'''graficamos los resultados accuracy/error'''
plot_history(history)

'''creamos la matriz de confusion'''
plot_matrix(X_test,y_test)

'''evaluamos el modelo con el set de test'''
test_loss0, test_acc0 = model.evaluate(X_test, y_test, verbose=False)

'''usamos el modelo con el set de test'''
best_model = load_model(model_file)
test_loss1, test_acc1 = best_model.evaluate(X_test, y_test, verbose=False)

'''guardamos los resulatos'''
with open(f'{resultados}std_{test_acc0}.txt',"w") as file:
    file.write(f'\nTest accuracy eval: {test_acc0}')
    file.write(f'\nTest loss eval: {test_loss0}')
    file.write(f'\nTest accuracy model: {test_acc1}')
    file.write(f'\nTest loss model: {test_loss1}')
    file.write(f'\n Entrenamientos: {n}')
    file.close()

new = datetime.now()
with open(f'{resultados}std_tiempo.txt',"w") as file:
    file.write(f'\ninicio de la ejecucion ltbdm: {now.day}-{now.month}-{now.year}  {now.hour}:{now.minute}:{now.second}')
    file.write(f'\nfin de la ejecucion ltbdm: {new.year}-{new.month}-{new.day}  {new.hour}:{new.minute}:{new.second}')
    file.close()
    
    
    
    
    
    
''':::::::::::::MODELO SCHDLR::::::::::::::::::'''
now = datetime.now()
print(f'\nINICIO DE EJECUCION SCHDLR: {now.day}-{now.month}-{now.year}  {now.hour}:{now.minute}:{now.second}')
resultados =f'{drive}/modelo_full_schdlr/'
datos = f'{drive}/full_database_project/'
sumary = f'{resultados}schdlr_model_summary.txt'
model_file = f'{resultados}schdlr_modelo.h5'

'''creamos la carpeta para guardar los resultados'''
if not os.path.exists(resultados):
    os.makedirs(resultados)

j=0
'''construimos el modelo a sintonizar'''
def build_model(hp):
    '''creamos el contenedor de la red LSTM'''
    model = keras.Sequential(name='model_LSTM')
    '''numero de neuronas y tamaño de la capa de entrada'''
    model.add(keras.layers.LSTM(hp.Choice('input_layer',values=[32,64]),return_sequences=True,input_shape=input_shape,name = 'input_layer'))
    '''iteramos en un rango de 1 a 10 capas ocultas'''
    for i in range (hp.Int('hidden_layers',min_value = 1, max_value=3)):
        model.add(keras.layers.LSTM(hp.Choice(f'hidden_layer_{i}',values=[32,64]),return_sequences=True,name=f'hidden_layer_{i}'))
        model.add(LeakyReLU(alpha=0.05,name=f'leakyrelu_{i}'))
        model.add(Dropout(0.2,name=f'dropout_{i}'))
    model.add(keras.layers.LSTM(hp.Choice(f'hidden_layer_{i+1}',values=[32,64]),dropout=0.2,name=f'hidden_layer_{i+1}'))
    model.add(LeakyReLU(alpha=0.05,name=f'leakyrelu_{i+1}'))
    model.add(Dropout(0.2,name=f'dropout_{i+1}'))
    '''capa de salida'''
    model.add(keras.layers.Dense(4, activation='softmax', name='output_layer_softmax'))
    model.compile(optimizer=Adam(0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    global j
    j+=1
    print(f'modelo: {j}')
    return model

'''instanciamos el sintonizador con los hyperparametros'''
tuner = kt.Hyperband(build_model,#Sintonizador hiperbanda
                      objective='val_accuracy',
                      max_epochs=epochs,
                      factor=3,
                      hyperband_iterations=1,
                      directory=resultados,
                      project_name="trial_ltbdm",
                      overwrite=True)

'''funcion para reducir la taza de aprendizaje'''
def scheduler(epoc, lr):
    if epoc <= 10:
        return lr
    else:
        return lr * tf.math.exp(-0.01)

'''procedemos con el ajuste'''
callback_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)
tuner.search(X_train, y_train, validation_data=(X_test, y_test),epochs=epochs, callbacks=[stop_early,callback_scheduler],use_multiprocessing=True,verbose=False)

'''extraemos los hiperparametros'''
best_hps= tuner.get_best_hyperparameters()[0]
model = tuner.hypermodel.build(best_hps)


model_checkpoint = ModelCheckpoint(filepath=model_file, monitor='val_accuracy',
                                    save_best_only=True, save_weights_only=False,mode='auto',verbose=False,period=1)
csv_logger = CSVLogger(f'{resultados}datos.csv', append=False, separator=',')
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                                    batch_size=64, epochs=epochs,callbacks=[csv_logger,
                                                                            model_checkpoint, callback_scheduler],use_multiprocessing=True,verbose=2)
'''visualizamos el modelo'''
model.summary(print_fn=myprint)

'''graficamos los resultados accuracy/error'''
plot_history(history)

'''creamos la matriz de confusion'''
plot_matrix(X_test,y_test)

'''evaluamos el modelo con el set de test'''
test_loss0, test_acc0 = model.evaluate(X_test, y_test, verbose=False)

'''usamos el modelo con el set de test'''
best_model = load_model(model_file)
test_loss1, test_acc1 = best_model.evaluate(X_test, y_test, verbose=False)

'''guardamos los resulatos'''
with open(f'{resultados}schdlr_{test_acc0}.txt',"w") as file:
    file.write(f'\nTest accuracy eval: {test_acc0}')
    file.write(f'\nTest loss eval: {test_loss0}')
    file.write(f'\nTest accuracy model: {test_acc1}')
    file.write(f'\nTest loss model: {test_loss1}')
    file.write(f'\n Entrenamientos: {j}')
    file.close()

new = datetime.now()
with open(f'{resultados}schdlr_tiempo.txt',"w") as file:
    file.write(f'\ninicio de la ejecucion schdlr: {now.day}-{now.month}-{now.year}  {now.hour}:{now.minute}:{now.second}')
    file.write(f'\nfin de la ejecucion schdlr: {new.year}-{new.month}-{new.day}  {new.hour}:{new.minute}:{new.second}')
    file.close()
