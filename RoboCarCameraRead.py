# -*- coding: utf-8 -*-
"""
Created on Fri May  8 04:57:00 2020

@author: Robomy
"""

import http.client
from PIL import Image
import io
import numpy as np
import cv2

host = "192.168.1.250"
port = 8080
argv = "/?action=snapshot"


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
    
    #show the image
    cv2.imshow('frame', open_cv_image)
    #wait each frame, press q to close
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()