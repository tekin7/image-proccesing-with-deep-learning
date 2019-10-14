# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 18:09:51 2019

@author: Lenovo-320
"""

import cv2
import numpy as np
import os
from random import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
#%% 

EGITIM_KLASORU = 'egitim'

RESIM_BOYUTU = 100
 
OGRENME_ORANI = 1e-3

MODEL_ADI = 'testt'
 
#%%
 
def etiket_olustur(img):
 obje_turu = img.split('.')[-3]
 if obje_turu == 'was':
     return np.array([1, 0, 0])
 elif obje_turu == 'dis':
     return np.array([0, 1, 0])
 elif obje_turu == 'ref':
     return np.array([0, 0, 1])

#%% 
 
def egitim_data_olustur():
    egitim_data=[]
    for img in tqdm(os.listdir(EGITIM_KLASORU)):
        label=etiket_olustur(img)
        path = os.path.join(EGITIM_KLASORU,img)
        img = cv2.resize(cv2.imread(path), (RESIM_BOYUTU ,RESIM_BOYUTU ))
        egitim_data.append([np.array(img),np.array(label)])
    shuffle(egitim_data)
    np.save('yeni_veri_olustur.npy',egitim_data)
    return egitim_data

#train_data = egitim_data_olustur()
#%%
from sklearn.model_selection import train_test_split
egitim_verisi = np.load('yeni_veri_olustur.npy', allow_pickle=True)
#test_verisi = np.load('test_data.npy')
X_egitim = np.array([i[0] for i in egitim_verisi]).reshape(-1, RESIM_BOYUTU, RESIM_BOYUTU, 3)
Y_egitim = [i[1] for i in egitim_verisi]

X_egitim, X_test, Y_egitim, Y_test= train_test_split(X_egitim, Y_egitim , test_size=0.15, random_state=2)

X_egitim=np.array(X_egitim)
Y_egitim=np.array(Y_egitim)
X_test=np.array(X_test)
Y_test=np.array(Y_test)

#%%
from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
image_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=.15,
    height_shift_range=.15,
    brightness_range=(1,2),
    horizontal_flip=True)

image_gen.fit(X_egitim, augment=True)
#%%
input_shape=(RESIM_BOYUTU, RESIM_BOYUTU, 3)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

cnn = Sequential()
cnn.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(BatchNormalization())

cnn.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(BatchNormalization())

cnn.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(BatchNormalization())

cnn.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(BatchNormalization())

cnn.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(BatchNormalization())

cnn.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(BatchNormalization())

cnn.add(Flatten())
cnn.add(Dropout(0.5))
cnn.add(Dense(16, activation='relu'))
cnn.add(BatchNormalization())


cnn.add(Dense(3, activation='softmax'))
#%% OLUŞTURULAN MİMARİ İLE DEEP LEARNING NETWORK (DNN) MODELİ OLUŞTURULMASI
import keras

cnn.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
cnn.summary()

#%% VERİLERLE EĞİTİM YAPILMASI
cnn.fit(X_egitim, Y_egitim, batch_size=56,
          epochs=25,
          verbose=1,
          validation_data=(X_test, Y_test))
#%%
TEST_KLASORU='test'
def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_KLASORU)):
        path = os.path.join(TEST_KLASORU,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path)
        img = cv2.resize(img, (RESIM_BOYUTU,RESIM_BOYUTU))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

tahmin_verisi = process_test_data()
#%%

ag_cikisi = cnn.predict([veri])[0]
 
if np.argmax(ag_cikisi) == 0:
    str_label = 'çamaşır makinesi'
elif np.argmax(ag_cikisi) == 1:
    str_label = 'bulaşık makinesi'
elif np.argmax(ag_cikisi) == 2:
    str_label = 'Buzdolabı'
camasir=round((ag_cikisi[0]*100),2)
bulasik=round((ag_cikisi[1]*100),2)
buzdolabi=round((ag_cikisi[2]*100),2)

print('camasir makinesi %',camasir)
print('bulasik makinesi %',bulasik)
print('buzdolabi %',buzdolabi)


#%%
from keras.models import load_model
#cnn.save(MODEL_ADI)
cnn=load_model(MODEL_ADI)
#%%
im='de.jpg'
im=cv2.imread(im)        
im=cv2.resize(im, (RESIM_BOYUTU,RESIM_BOYUTU))
veri = im.reshape(-1,RESIM_BOYUTU, RESIM_BOYUTU, 3)
#%% confusıon matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix 
Y_pred=cnn.predict(X_test)
Y_pred_classes =np.argmax(Y_pred, axis=1)
Y_true=np.argmax(Y_test, axis=1)

confusion_mtx=confusion_matrix(Y_true, Y_pred_classes)

f,ax=plt.subplots(figsize=(8,8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap='Greens', linecolor='gray', fmt= '.1f', ax=ax )

plt.xlabel("predicted label")
plt.ylabel("True label")
plt.title("confusion matrix")
plt.show()
#%%
cvscores = []

scores = cnn.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (cnn.metrics_names[1], scores[1]*100))
cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

