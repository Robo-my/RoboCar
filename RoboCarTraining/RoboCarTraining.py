# -*- coding: utf-8 -*-
"""
Created on Fri May  8 06:53:26 2020

@author: Robomy
"""

from RoboTrain import trainInit, train,checkData


'''it return images in required format and 
number of images use in each step'''
train_generator, step_size_train = trainInit()

'''train the images and save a model
epochs means how many times training need to done'''
train(train_generator, step_size_train,  epochs=5)
