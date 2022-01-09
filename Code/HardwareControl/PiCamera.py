import picamera
import time

# take pic
camera = picamera.PiCamera()

      
camera.capture('motion.jpg')
print('camera take the picture')