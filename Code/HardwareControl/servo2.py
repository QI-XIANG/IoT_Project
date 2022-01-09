# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT

"""Simple test for a standard servo on channel 0 and a continuous rotation servo on channel 1."""
import time
from adafruit_servokit import ServoKit

# Set channels to the number of servo channels on your kit.
# 8 for FeatherWing, 16 for Shield/HAT/Bonnet.
kit = ServoKit(channels=16)
right = 1
left = 2
counter = 1
while counter <= 15:
    kit.continuous_servo[0].throttle = 0.5
    time.sleep(0.08*right)
    kit.continuous_servo[0].throttle = 0
    print("sleeep")
    time.sleep(2)
    kit.continuous_servo[0].throttle = -0.5
    time.sleep(0.08*left)
    kit.continuous_servo[0].throttle = 0
    print("sleeep")
    time.sleep(2)
    temp = right
    right = left
    left = temp
    counter += 1
    

kit.continuous_servo[0].throttle = 0