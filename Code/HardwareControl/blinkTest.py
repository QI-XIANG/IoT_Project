#讓紅色及綠色LED各閃10次，每次間隔0.5秒
import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)

#設定LED pin變數
LED_R    = 32
LED_G    = 36
LED_Y    = 38
counter = 0

#設定為輸出
GPIO.setup(LED_R,GPIO.OUT)
GPIO.setup(LED_Y,GPIO.OUT)
GPIO.setup(LED_G,GPIO.OUT)

#迴圈10次
while(counter < 20):
        GPIO.output(LED_R,GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(LED_R,GPIO.LOW)
        time.sleep(0.5)
        GPIO.output(LED_Y,GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(LED_Y,GPIO.LOW)
        time.sleep(0.5)
        GPIO.output(LED_G,GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(LED_G,GPIO.LOW)
        time.sleep(0.5)
        counter = counter + 1
GPIO.output(LED_R,GPIO.LOW)
GPIO.cleanup()