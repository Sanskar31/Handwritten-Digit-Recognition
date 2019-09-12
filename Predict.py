# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 19:47:19 2019

@author: Sanskar
"""
import numpy
import cv2
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense

img_width, img_height = 28, 28

def create_model():
  model = Sequential()

  model.add(Convolution2D(16, 5, 5, activation='relu', input_shape=(img_width, img_height, 3)))
  model.add(MaxPooling2D(2, 2))

  model.add(Convolution2D(32, 5, 5, activation='relu'))
  model.add(MaxPooling2D(2, 2))

  model.add(Flatten())
  model.add(Dense(1000, activation='relu'))

  model.add(Dense(10, activation='softmax'))

  model.summary()

  return model


img = cv2.imread('Live.png')
img = cv2.resize(img, (img_width, img_height))
model = create_model()
model.load_weights('model_k_validation_final.h5')
arr = numpy.array(img).reshape((img_width,img_height,3))
arr = numpy.expand_dims(arr, axis=0)
prediction = model.predict(arr)[0]
bestclass = ''
bestconf = -1
for n in [0,1,2,3,4,5,6,7,8,9]:
	if (prediction[n] > bestconf):
		bestclass = str(n)
		bestconf = prediction[n]
print('I think this digit is a ' + bestclass)
