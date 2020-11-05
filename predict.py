#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import preprocess_input
from keras.models import Sequential
from glob import glob
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


class maleFemale:
    def __init__(self, filename):
        self.filename = filename

    def predictionmaleFemale(self):
        # load model
        model = keras.models.load_model('TF_Resnet50.h5')

        # summarize model
        # model.summary()
        imagename = self.filename
        # load an image from file
        image = load_img(imagename, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        yhat = model.predict(image)
        # print(yhat)
        # print(np.argmax(yhat[0]))

        classification = [np.argmax(yhat[0])]
        # print(classification)
        if classification == [0]:
            prediction = 'men'
            # print(prediction)
            return [{"image": prediction}]
        elif classification == [1]:
            prediction = 'women'
            # print(prediction)
            return [{"image": prediction}]

