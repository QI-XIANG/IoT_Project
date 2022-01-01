# IoT_Project
2021 IoT Final Project

I will update some content here soon !

2021/12/08 update linebot template 

---

# My Little Skynet

###### tags: `IoT`

> This is the introduction of the IoT project. 🎅

"My Little Skynet" is the final project of the IoT course at NCU, Taiwan. This project may still have many bugs or defects to fix and improve. So it is very welcome for you to issue me on my Github. ^^

## :book: Story Behind The Project

### Why named this project "My Little Skynet" ?

In China, Skynet is a huge social monitor system. This system built with “Face Recognition”、“Big Data”、“IoT” and other technologies successfully forms the social security chain. This inspired me to name the project, so I finally name it “My Little Skynet”.

This project is a simple implementation of the real-world Skynet.

---

## :bulb: Possible User Cases

* Assumptions:
    * The image recognition model can only identify two types of objects like family and stranger and does not identify other objects.
    * Do not provide the video live streaming, that is user can't watch the in-time video with the application interface. (User can only see the image captured by the camera during the rotation of servo motor.) 
    * While using the device, the room or environment should have a weak light source at least. Do not use it without light, or it will be hard for the device to identify an image. 

* Situation: Check whether there is any stranger.  
     * With Stranger 
        1. Raspberry Pi will keep getting images from the connected camera.
        2. Raspberry Pi will use the image and recognition model to identify if there exists any stranger. 
        3. After identifying by the Raspberry Pi there exists a stranger. 
        4. Three things will happen:
            * The buzzer will keep ringing loudly.
            *  The light‑emitting diode will keep shining with red light.
            *  Raspberry Pi will use Line to send a hint message to inform the user that a stranger has entered the room.
        5. User can interact with Line Bot and press down the "taking photo" button to ask Raspberry Pi for taking pictures and return them.
        6. Raspberry Pi will use Line to send the image gotten from the connected camera every 5 seconds for 1 minute with 12 images in total. 
        7. User can press down the "reset" button to reset the status of the device.
     
     * Without Stranger

![](https://imgur.dcard.tw/VNQcyu8h.jpg)
