import time
from adafruit_servokit import ServoKit

# Set channels to the number of servo channels on your kit.
# 8 for FeatherWing, 16 for Shield/HAT/Bonnet.
kit = ServoKit(channels=16)

class servoMotion():
    def __init__(self, right, left):
        self.right = right
        self.left = left

    def getRight(self):
        return self.right

    def getLeft(self):
        return self.left

    def turnRight(self):
        if self.right >= 1:
            kit.continuous_servo[0].throttle = 0.5
            time.sleep(0.2)
            kit.continuous_servo[0].throttle = 0
            time.sleep(1)
            print("sleeep")
            self.right = self.right-1
            self.left = self.left+1
            print('turn right')
        else:
            print("You have no chance to turn right")
    
        print(self.right,self.left)

    def turnLeft(self):
        if self.left >= 1:
            kit.continuous_servo[0].throttle = -0.5
            time.sleep(0.2)
            kit.continuous_servo[0].throttle = 0
            time.sleep(1)
            print("sleeep")
            self.left -= 1
            self.right += 1
            print('turn left')
        else:
            print("You have no chance to turn left")
        
        print(self.right,self.left)

    def resetStatus(self):
        if self.right > 1:
            kit.continuous_servo[0].throttle = 0.5
            time.sleep(0.2)
            kit.continuous_servo[0].throttle = 0
            time.sleep(1)
            print("sleeep")
            print('reset turn right')
            self.right -= 1
            self.left += 1
        elif self.left > 1:
            kit.continuous_servo[0].throttle = -0.5
            time.sleep(0.2)
            kit.continuous_servo[0].throttle = 0
            time.sleep(1)
            print("sleeep")
            print('reset turn left')
            self.left -= 1
            self.right += 1;
    