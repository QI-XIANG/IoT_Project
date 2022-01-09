import picamera
import time
#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.models import load_model
from keras.models import load_model
from keras.preprocessing import image
#import cv2
import numpy as np

camera = picamera.PiCamera()
counter = 1
while counter <= 5:
# take pic
    pic = 'fit'+str(counter)+'.jpg'
    camera.capture(pic)
    print('camera take the picture')
    print(pic)
    #Prediction('fit.jpg')
    counter += 1       