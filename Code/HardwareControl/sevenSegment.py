import RPi.GPIO as GPIO
import time
import os, sys
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
#setup output pins
GPIO.setup(18, GPIO.OUT)      
GPIO.setup(23, GPIO.OUT)      
GPIO.setup(24, GPIO.OUT)      
GPIO.setup(25, GPIO.OUT)      
GPIO.setup(8, GPIO.OUT)      
GPIO.setup(7, GPIO.OUT)      
GPIO.setup(1, GPIO.OUT)

p0 = GPIO.PWM(18,50)
p1 = GPIO.PWM(23,50)
p2 = GPIO.PWM(24,50)
p3 = GPIO.PWM(25,50)
p4 = GPIO.PWM(8,50)
p5 = GPIO.PWM(7,50)
p6 = GPIO.PWM(1,50)
p1.start(0)
p2.start(0)
p3.start(0)
p4.start(0)
p5.start(0)
p6.start(0)
p0.start(0)

p = [p0,p1,p2,p3,p4,p5,p6]

#define 7 segment digits
digitclr=[0,0,0,0,0,0,0]
digit0=[1,1,1,1,1,1,0]
digit1=[0,1,1,0,0,0,0]
digit2=[1,1,0,1,1,0,1]
digit3=[1,1,1,1,0,0,1]
digit4=[0,1,1,0,0,1,1]
digit5=[1,0,1,1,0,1,1]
digit6=[1,0,1,1,1,1,1]
digit7=[1,1,1,0,0,0,0]
digit8=[1,1,1,1,1,1,1]
digit9=[1,1,1,0,0,1,1]
gpin=[18,23,24,25,8,7,1]
#routine to clear and then write to display
def digdisp(digit,light):
    for x in range (0,7):
        p[x].ChangeDutyCycle(digitclr[x]*light*10)
    for x in range (0,7):
        p[x].ChangeDutyCycle(digit[x]*light*10)
        #GPIO.output(gpin[x], digit[x])
#routine display digit from 0 to 9
flag = True
counter = 1
while flag:
    digdisp(digit0,1)
    time.sleep(1)
    digdisp(digit1,2)
    time.sleep(1)
    digdisp(digit2,3)
    time.sleep(1)
    digdisp(digit3,4)
    time.sleep(1)
    digdisp(digit4,5)
    time.sleep(1)
    digdisp(digit5,6)
    time.sleep(1)
    digdisp(digit6,7)
    time.sleep(1)
    digdisp(digit7,8)
    time.sleep(1)
    digdisp(digit8,9)
    time.sleep(1)
    digdisp(digit9,10)
    time.sleep(1)
    digdisp(digitclr,0)
    time.sleep(1)
    choice = input('y/n')
    if choice == 'n':
        flag = False
#tidy up
p.stop()
GPIO.cleanup()
import sys
sys.exit()