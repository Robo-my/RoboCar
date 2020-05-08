# -*- coding: utf-8 -*-
"""
Created on Fri May  8 08:16:00 2020

@author: Robomy
"""



import http.client
from PIL import Image
import io
import numpy as np
import cv2
import requests
from RoboTrain import loadModel,load_image

host = "192.168.1.250"
port = 8080
actionport = 8000
argv = "/?action=snapshot"
BASE_URL = 'http://' + host + ':' + str(actionport) + '/'

# action command to Robocar
def action(cmd):
    url = BASE_URL + 'run/?action=' + cmd
    requests.get(url=url)

# make tyre stright 
action('fwstraight')  
# make camera stright 
action('camready')


#load the model
model = loadModel()

while(True):

    #read image form http
    http_data = http.client.HTTPConnection(host, port)
    http_data.putrequest('GET', argv)
    http_data.putheader('Host', host)
    http_data.putheader('User-agent', 'python-http.client')
    http_data.putheader('Content-type', 'image/jpeg')
    http_data.endheaders()

    #covert to image format
    returnmsg = http_data.getresponse()
    image = Image.open(io.BytesIO(returnmsg.read()))
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1]
    
    new_image = load_image(open_cv_image)
    pred = model.predict(new_image)
    
    #check for good accuracy
    if(pred[0][0] > 0.99  or pred[0][1] > 0.99):
            if(pred[0][0] > pred[0][1]):
                #if class 1 wheel left
                action('fwleft')  
            else:
                #if class 2 wheel right
                action('fwright')
    else:
        action('fwstraight')                
                
            #show the image
    cv2.imshow('frame', open_cv_image)
    
    
    #wait each frame, press q to close
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()