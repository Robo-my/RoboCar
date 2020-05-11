# -*- coding: utf-8 -*-

from PIL import Image
from keras.preprocessing import image as KerasImage
import tensorflow as tf
import cv2
import numpy as np
def load_image(frame):


        frame = cv2.resize(frame,(224,224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        # (height, width, channels)
        img_tensor = KerasImage.img_to_array(img)
        # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        # imshow expects values in the range [0, 1]
        img_tensor /= 255.
        return img_tensor   

interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(open_cv_image):
    new_image = load_image(open_cv_image)
    interpreter.set_tensor(input_details[0]['index'], new_image)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])
    return(pred)