from typing import Counter
from ImageClassification.Predict import Prediction
import picamera
import time


counter = 1
while counter <= 5:
# take pic
    camera = picamera.PiCamera()
    camera.capture('fit.jpg')
    print('camera take the picture')
    Prediction('fit.jpg')
    counter += 1                             



