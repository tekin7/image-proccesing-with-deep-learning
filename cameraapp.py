import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
RESIM_BOYUTU = 100

MODEL_ADI = 'testt'
#%%
from keras.models import load_model
#cnn.save(MODEL_ADI)
cnn=load_model(MODEL_ADI)

#%%
tahmin_verisi = np.load('test_data.npy')
 
for no, veri in enumerate(tahmin_verisi[:]):
 
 resim_no = veri[1]
 resim_verisi = veri[0]
 
 orig = resim_verisi
 veri = resim_verisi.reshape(-1,RESIM_BOYUTU, RESIM_BOYUTU, 1)
 ag_cikisi = cnn.predict([veri])[0]
 
 if np.argmax(ag_cikisi) == 0:
     str_label = 'çamaşır makinesi'
 elif np.argmax(ag_cikisi) == 1:
     str_label = 'bulaşık makinesi'
 elif np.argmax(ag_cikisi) == 2:
     str_label = 'Buzdolabı'
     
#%%
scores = cnn.evaluate(X_test, y_test, verbose=0)
#%%
video = cv2.VideoCapture(0)

while True:
    _, frame = video.read()

    im = Image.fromarray(frame, 'RGB')

    im = im.resize((RESIM_BOYUTU, RESIM_BOYUTU))

    veri = np.array(im)
    veri = np.expand_dims(veri, axis=0)

    prediction = cnn.predict([veri])[0]

    if np.argmax(prediction) == 0:
        str_label = 'çamaşır makinesi'
    elif np.argmax(prediction) == 1:
        str_label = 'bulaşık makinesi'
    elif np.argmax(prediction) == 2:
        str_label = 'buzdolabi'
    camasir = round((prediction[0] * 100), 2)
    bulasik = round((prediction[1] * 100), 2)
    buzdolabi = round((prediction[2] * 100), 2)
    print('camasir makinesi %', camasir)
    print('bulasik makinesi %', bulasik)
    print('buzdolabi %', buzdolabi)
    cv2.imshow("Capturing", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()

