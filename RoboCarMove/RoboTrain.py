# -*- coding: utf-8 -*-
"""
Created on Fri May  8 06:46:53 2020

@author: User
"""
import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
import numpy as np
from keras.optimizers import Adam
from keras.models import load_model
import os
import cv2
from PIL import Image

def checkData():
    class1List = os.listdir('images/class1')
    class2List = os.listdir('images/class2')
    if(len(class1List) > 29 and len(class2List) > 29):
        if(len(class1List) > 100 or len(class2List) > 100):
            print('Maximum 100 images in both classes')
            return False
        else:
            return True
    else:
        print('Atleast 30 images required in both classes')
           
        return False


# imports the mobilenet model and discards the last 1000 neuron layer.
base_model = MobileNet(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
# we add dense layers so that the model can learn more complex functions and classify for better results.
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)  # dense layer 2
x = Dense(512, activation='relu')(x)  # dense layer 3
# final layer with softmax activation
preds = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)

for layer in model.layers[:20]:
    layer.trainable = False
for layer in model.layers[20:]:
    layer.trainable = True

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)  # included in our dependencies





def trainInit():
    train_generator = train_datagen.flow_from_directory('images',
                                                        target_size=(224, 224),
                                                        color_mode='rgb',
                                                        batch_size=32,
                                                        class_mode='categorical',
                                                        shuffle=True)

    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    step_size_train = train_generator.n//train_generator.batch_size
    return train_generator, step_size_train


def train(train_generator, step_size_train, epochs):

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=step_size_train,
                        epochs=epochs)
    model.save('model.h5')
    
def loadModel():
    keras.applications.mobilenet.MobileNet()

    base_model = MobileNet(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # we add dense layers so that the model can learn more complex functions and classify for better results.
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)  # dense layer 2
    x = Dense(512, activation='relu')(x)  # dense layer 3
    # final layer with softmax activation
    preds = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=preds)
    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # model.save('../model/my_model.h5')
    model = load_model('model.h5')

    return model    

def load_image(frame):


        frame = cv2.resize(frame,(150,150))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        # (height, width, channels)
        img_tensor = image.img_to_array(img)
        # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        # imshow expects values in the range [0, 1]
        img_tensor /= 255.
        return img_tensor    