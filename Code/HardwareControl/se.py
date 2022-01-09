import time
from adafruit_servokit import ServoKit

# Set channels to the number of servo channels on your kit.
# 8 for FeatherWing, 16 for Shield/HAT/Bonnet.
kit = ServoKit(channels=16)

right = 1
left = 1

def turnRight(rightChance,leftChance):
    if rightChance >= 1:
        kit.continuous_servo[0].throttle = 0.5
        time.sleep(0.2)
        kit.continuous_servo[0].throttle = 0
        time.sleep(1)
        print("sleeep")
        rightChance = rightChance-1
        leftChance = leftChance+1
        print('turn right')
    else:
        print("You have no chance to turn right")
    
    print(rightChance,leftChance)
    return rightChance,leftChance

def turnLeft(rightChance,leftChance):
    if leftChance >= 1:
        kit.continuous_servo[0].throttle = -0.5
        time.sleep(0.2)
        kit.continuous_servo[0].throttle = 0
        time.sleep(1)
        print("sleeep")
        leftChance -= 1
        rightChance += 1
        print('turn left')
    else:
        print("You have no chance to turn left")
        
    print(rightChance,leftChance)
    return rightChance,leftChance

def resetStatus(rightChance,leftChance):
    if rightChance > 1:
        kit.continuous_servo[0].throttle = 0.5
        time.sleep(0.2)
        kit.continuous_servo[0].throttle = 0
        time.sleep(1)
        print("sleeep")
        print('reset turn right')
        rightChance -= 1
    elif leftChance > 1:
        kit.continuous_servo[0].throttle = -0.5
        time.sleep(0.2)
        kit.continuous_servo[0].throttle = 0
        time.sleep(1)
        print("sleeep")
        print('reset turn left')
        leftChance -= 1
    return rightChance,leftChance
    
'''right,left = turnRight(right,left)
right,left = turnRight(right,left)
right,left = turnRight(right,left)
right,left = turnLeft(right,left)
right,left = turnLeft(right,left)
right,left = turnLeft(right,left)
right,left = turnRight(right,left)
right,left = resetStatus(right,left)'''
    