# Practical Internet of Things 

![](https://badgen.net/github/watchers/QI-XIANG/NCU_IoT_Project) ![](https://badgen.net/github/commits/QI-XIANG/NCU_IoT_Project) ![](https://badgen.net/github/last-commit/QI-XIANG/NCU_IoT_Project) ![](https://badgen.net/github/license/QI-XIANG/NCU_IoT_Project)

| 課程名稱: | 課程時間 | 授課教授 |
| -------- | -------- | -------- |
| 物聯網實務應用     | 2021年9月~隔年1月     | 柯士文教授     |

[Full Document](https://hackmd.io/@qixiang1009/SJ-8ugRoY/https%3A%2F%2Fhackmd.io%2Fj37E31Q8QgC-W-Iine0kTg%3Fview%23My-Little-Skynet)

# My Little Skynet

> This is the introduction of the IoT project. 🎅

"My Little Skynet" is the final project of the IoT course at NCU, Taiwan. This project may still have many bugs or defects to fix and improve. So it is very welcome for you to issue me on my Github.

# Overview

## :book: Story Behind The Project

> This is the introduction of the IoT project. 🎅

"My Little Skynet" is the final project of the IoT course at NCU, Taiwan. This project may still have many bugs or defects to fix and improve. So it is very welcome for you to issue me on my Github.


### Why named this project "My Little Skynet" ?

In China, Skynet is a huge social monitor system. This system built with “Face Recognition”、“Big Data”、“IoT” and other technologies successfully forms the social security chain. This inspired me to name the project, so I finally name it “My Little Skynet”.

![](https://i.imgur.com/ltc4lTk.jpg)

This project is a simple implementation of the real-world Skynet.

This device will inform the user by sending a Line message if any stranger or weird thing is entering the room or other places.

**Due to a lack of time to fix the TensorFlow package installation problem,  stranger detection will not be implemented in the IoT project this time.** 

**The model was trained under the newest Tensorflow, but I can't download the newest version of Tensorflow in Rasberry pi 3 model B without enough resources to finish the download.**

**Each time I download it to 99%, the download process is killed by Raspbian OS.**


## :bulb: Possible User Cases

### Original (Under Ideal Condition)


* Assumptions:
    * The image recognition model can only identify two types of objects like family and stranger and does not identify other objects.
    * Do not provide the video live streaming, that is user can't watch the in-time video with the application interface. (User can only see the image captured by the camera during the rotation of servo motor.) 
    * While using the device, the room or other places should have a weak light source at least. Do not use it without light, or it will be hard for the device to identify an image. 

* Situation:
 Check whether there is any stranger.
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
        1. Raspberry Pi will keep getting images from the connected camera.
        2. Raspberry Pi will use the image and recognition model to identify if there exists any stranger. 
        3. After identifying by the Raspberry Pi there does not exist any stranger. 
        4. Only one thing will happen:
            *  The light‑emitting diode will keep lighting with green light.
        5. User can interact with Line Bot and press down the "taking photo" button to ask Raspberry Pi for taking pictures and return them.
        6. Raspberry Pi will use Line to send the image gotten from the connected camera every 5 seconds for 1 minute with 12 images in total. 

### Implementation for Real

* Assumptions:
    * Do not provide the video live streaming, that is user can't watch the in-time video with the application interface. (User can only see the image captured by the camera during the rotation of servo motor.) 
    * While using the device, the room or other places should have a weak light source at least. Do not use it without light, or it will be hard for you to look the image. 

* Normal Situation:

1. User can change three kinds of the degree to take a different image.
2. Line Bot will return the image captured by the camera immediately.
3. User can easily know what time the image was taken.
4. User can reset the servo motor.


## :thought_balloon: Required Components

| Name              | Quantity                    | 
| ----------------- |:-----------------------  |
| Raspberry Pi 3 model B      | 1   |
|32G SD card| 1 | 
| Breadboard | 1    | 
|          5V Traffic Light LED Display Module        |      1                   | 
| Dupont Line         | many   |  
| Raspberry Pi Camera Moudule V2        | 1  |  
|Raspberry Pi Camera Module V2 Case| 1| 
| MG996R 55g Metal Gear Torque Digital Servo Motor   | 1 | 
| ~~Buzzer~~  | 1 |  
| ~~Intel® Neural Compute Stick 2~~ | 1 |
| Adhesive Tape  | 1 |
| Carton| 1 |
|~~GPIO Expansion T Board for Raspberry Pi~~|1|
|~~Rainbow Cable~~|1|
|Adafruit PCA9685 16-Channel Servo Driver|1|
|AC Power Adaptor|1|

---

# Prerequisites

## :books: Line Bot

>First of all, you should have a Line account before creating your Line Bot. 

### [Line Developers](https://developers.line.biz/en/)

1. Go to the Line Developers Website and log in.

![](https://i.imgur.com/dPjDCnj.jpg)

2. After login, you can see the following page.

![](https://i.imgur.com/6zx1XPf.png)

3. Click the **"Create"** button to create a new provider.

    ![](https://i.imgur.com/LqKR4D5.jpg)

    * Enter the provider name you like. ![](https://i.imgur.com/5YSBYPc.png)

    * You can see the provider appear on the left-hand side. ![](https://i.imgur.com/KvH1FF4.png)

4. Create a Messaging API Channel.

    ![](https://i.imgur.com/cW8Ti90.jpg)

    * Here you can edit your channel's detail information. It is easy for you to explore it by yourself. ![](https://i.imgur.com/TuCyYFy.png)
    * You will see the following graph after you finish editing the channel information. ![](https://i.imgur.com/YjN8OjP.png)
 

### [LINE Messaging API SDK for Python](https://github.com/line/line-bot-sdk-python)

In this repository, you can get some instructions and the Line Bot Template.

![](https://i.imgur.com/iFtJ1rf.png)

#### **Line Bot Template (Python)**

> Please check you have installed all the needed packages.

```python=
from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)

app = Flask(__name__)

line_bot_api = LineBotApi('YOUR_CHANNEL_ACCESS_TOKEN')
handler = WebhookHandler('YOUR_CHANNEL_SECRET')


@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=event.message.text))


if __name__ == "__main__":
    app.run()
```

### Webhook

 > A webhook in web development is a method of augmenting or altering the behavior of a web page or web application with custom callbacks. -- Wikipedia 

![](https://i.imgur.com/01en1fg.png)

The message the user sends can be passed by the LINE Platform and sends it to the Bot Server. Bot Server can also proactively use LINE Platform broadcasting message to the user.

Webhook is placed between the LINE Platform and Bot Server, it can use Messaging API to make both of them interact with each other when receiving different kinds of webhook events.

* **Two important things we need to make webhook work:**
    1. Channel Secret
        * A unique secret key you can use to grant an app access to your channel
        ![](https://i.imgur.com/XHmE3MZ.png)
    2. Channel Access Token
        * Channel Access Tokens as a means of authentication for channels
        ![](https://i.imgur.com/TVrC24q.png)

With webhook, we can simply customize the messages sent to the end user.

Next, we will use Ngrok to start localhost and make it work as our webhook.


### [Ngrok](https://ngrok.com)

> **What is ngrok?**
Ngrok exposes local servers behind NATs and firewalls to the public internet over secure tunnels.

Here you can [download](https://ngrok.com/download) the `ngrok.exe`.

![](https://i.imgur.com/qqpDxiQ.png)

**Why do we need to set a localhost server?**

We need a localhost to provide the webhook URL before sending customized message.

![](https://i.imgur.com/0oVJ6PV.png)

**How to use Ngrok?**

Run the ngrok.exe and key in `ngrok http 80`. 
It will expose a web server on port 80 of your local machine to the Internet.

![](https://i.imgur.com/HECnFR8.png)

![](https://i.imgur.com/OCodzUq.png)

### Reference

https://medium.com/@justinlee_78563/line-bot-%E7%B3%BB%E5%88%97%E6%96%87-%E4%BB%80%E9%BA%BC%E6%98%AF-webhook-d0ab0bb192be

https://ngrok.com/product

https://developers.line.biz/en/docs/messaging-api/line-bot-sdk/#messaging-api-sdks

https://github.com/line/line-bot-sdk-python

https://developers.line.biz/en/

https://www.webfx.com/tools/emoji-cheat-sheet/

## :sailboat: CNN Model

> CNN is the subset of deep learning, It is similar to the basic neural network. 
> 
> CNN is a type of neural network model which allows working with the images and videos, CNN takes the image’s raw pixel data, trains the model, then extracts the features automatically for better classification.

![](https://i.imgur.com/UmkGyyH.png)

### Image Classification Models: Little Data

In this project, we will only use little data (less than 2000 images) to train a powerful image classification model. 

Following Keras features will be used:

* `fit`: training Keras a model using Python data generators
* `ImageDataGenerator`: real-time data augmentation
...and more.

#### Setup: Directory Structure

```
C:.
│  306px-Rimuru.png
│  first_try.h5
│  GG.png
│  GG_edit_0_145.png
│  ImagePreProcessing.ipynb
│  model.h5
│  slime_edit_0_41.jpeg
│  tree.txt
│  
├─.ipynb_checkpoints
│      ImagePreProcessing-checkpoint.ipynb
│      
├─Test
│  ├─GG
│  │      GG.png
│  │      GG_edit_0_145.png
│  │      GG_edit_0_157.png
│  │      GG_edit_0_160.png
│  │      GG_edit_0_167.png
│  │      GG_edit_0_183.png
│  │      ...
│  │      
│  └─Slime
│         306px-Rimuru.png
│         slime.jpg
│         v_119519132_m_601_480_270.jpg
│         ...
│          
└─Train
    ├─GG
    │     GG_edit_0_1060.png
    │     GG_edit_0_1074.png
    │     GG_edit_0_1077.png
    │     GG_edit_0_1078.png
    │     GG_edit_0_1103.png
    │     GG_edit_0_1106.png
    │     GG_edit_0_1110.png
    │     GG_edit_0_1132.png
    │     GG_edit_0_1158.png
    │     GG_edit_0_1229.png
    │     ...
    │      
    └─Slime
          slime_edit_0_1.jpeg
          slime_edit_0_1002.jpeg
          slime_edit_0_1035.jpeg
          slime_edit_0_1055.jpeg
          slime_edit_0_1081.jpeg
          slime_edit_0_1090.jpeg
          slime_edit_0_1091.jpeg
          slime_edit_0_1106.jpeg
          slime_edit_0_1125.jpeg
          slime_edit_0_1128.jpeg
          ...        
```

#### Data Pre-Processing & Augmentation
In order to make the most of our few training examples, we will "augment" them via a number of random transformations, so that our model would never see twice the exact same picture. This helps prevent overfitting and helps the model generalize better.

In Keras this can be done via the keras.preprocessing.image.ImageDataGenerator class. This class allows you to:

* Configure random transformations and normalization operations to be done on your image data during training
* Instantiate generators of augmented image batches (and their labels) via .flow(data, labels) or .flow_from_directory(directory). These generators can then be used with the Keras model methods that accept data generators as inputs, fit and predict.

Example:

```python=
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
```

* `rotation_range`: value in degrees (0-180), a range within which to randomly rotate pictures
* `width_shift_range`、`height_shift_range`: ranges (as a fraction of total width or height) within which to randomly translate pictures vertically or horizontally
* `shear_range`: randomly applying shearing transformations
* `zoom_range`: randomly zooming inside pictures
* `horizontal_flip`: randomly flipping half of the images horizontally --relevant when there are no assumptions of horizontal assymetry (e.g. real-world pictures).
* `fill_mode`: the strategy used for filling in newly created pixels, which can appear after a rotation or a width/height shift.

Now let's start generating some pictures using this tool and save them to a temporary directory, so we can get a feel for what our augmentation strategy is doing : 

```python=
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = load_img('Test/Slime/slime.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='Train/Slime', save_prefix='slime', save_format='jpeg'):
    i += 1
    if i > 20: #only produce 501 images
        break  # otherwise the generator would loop indefinitely
```

Here's what we get -- this is what our data augmentation strategy looks like.

![](https://i.imgur.com/i9mften.jpg)

#### Training Convnet From Small DataSet

Full code for this project's model:

```python=
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3),input_shape=(150,150,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), padding='same'))

model.add(Conv2D(32, (3, 3),input_shape=(150,150,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), padding='same'))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2), padding='same'))
```

Stick two fully-connected layers. End the model with a single unit and a sigmoid activation, which is perfect for a binary classification. Use the binary_crossentropy as loss function to train the model.

```python=
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
```


Use `.flow_from_directory()` to generate batches of image data (and their labels) directly from our jpgs in their respective folders.

```python=
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'Train/',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=32,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'Test/',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
```

Use these generators to train the model. It's faster to run this model on GPU if you don't like time-consuming work.

```python=
model.fit(
        train_generator,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=800)

model.save('model.h5')
model.save_weights('first_try.h5')  # always save your weights after training or during training
```
This approach gets a validation accuracy of 0.9602-0.9826 after 5 epochs.

![](https://i.imgur.com/In8Mqe3.png)

#### Image Classification

In this section, we will use the model trained above to do image classification.

* **Class 0 : GG**
* **Class 1 : Slime**

```python=
from keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np

model = load_model('model.h5')
model.load_weights('first_try.h5')

model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

img_width, img_height = 150, 150

# predicting images
img = image.load_img('GG.png', target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes)

# predicting multiple images at once
img = image.load_img('slime_edit_0_41.jpeg', target_size=(img_width, img_height))
y = image.img_to_array(img)
y = np.expand_dims(y, axis=0)

# pass the list of multiple images np.vstack()
images = np.vstack([x, y])
classes = model.predict(images, batch_size=32).astype("int32")

for kind in classes:
    print(kind,end='')
```
![](https://i.imgur.com/AV0XSX0.png)

### Reference

https://www.analyticsvidhya.com/blog/2021/06/create-convolutional-neural-network-model-and-optimize-using-keras-tuner-deep-learning/

https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

https://keras-cn.readthedocs.io/en/latest/legacy/blog/image_classification_using_very_little_data/#_1

https://stackoverflow.com/questions/9518646/tree-view-of-a-directory-folder-in-windows

https://stackoverflow.com/questions/41651628/negative-dimension-size-caused-by-subtracting-3-from-1-for-conv2d

https://stackoverflow.com/questions/49079115/valueerror-negative-dimension-size-caused-by-subtracting-2-from-1-for-max-pool

https://www.tensorflow.org/api_docs/python/tf/keras/Model#methods_2

https://stackoverflow.com/questions/43469281/how-to-predict-input-image-using-trained-model-in-keras

https://stackoverflow.com/questions/64180817/typeerror-fit-generator-got-an-unexpected-keyword-argument-nb-val-samples/64203880

https://stackoverflow.com/questions/63359321/type-error-fit-generator-got-an-unexpected-keyword-argument-samples-per-epoc

https://keras.io/zh/models/model/

---

# Startup

## :book: 1. Python Packages Requirement

| Package  | Version |
| -------- | -------- |
|absl-py  |                        1.0.0      |
Adafruit-Blinka               |   6.17.0     |
adafruit-circuitpython-busdevice |5.1.1      
adafruit-circuitpython-motor     |3.3.1      
adafruit-circuitpython-pca9685  | 3.3.9      
adafruit-circuitpython-register  |1.9.6      
adafruit-circuitpython-servokit  |1.3.6      
Adafruit-PlatformDetect          |3.18.0     
Adafruit-PureIO                  |1.1.9      
aiohttp                          |3.8.1      
aiosignal                        |1.2.0      
asn1crypto                       |0.24.0     
astor                            |0.8.1      
astroid                          |2.1.0      
asttokens                        |1.1.13     
async-timeout                    |4.0.2      
asynctest                        |0.13.0     
attrs                            |21.4.0     
automationhat                    |0.2.0      
beautifulsoup4                   |4.7.1      
blinker                          |1.4        
blinkt                           |0.1.2      
buttonshim                       |0.0.2      
cached-property                  |1.5.2      
cachetools                       |4.2.4      
Cap1xxx                          |0.1.3      
certifi                          |2018.8.24  
chardet                          |3.0.4      
charset-normalizer               |2.0.10     
Click                            |7.0        
colorama                         |0.3.7      
colorzero                        |1.1        
cookies                          |2.2.1      
cryptography                     |2.6.1      
cupshelpers                      |1.0        
docutils                         |0.14       
drumhat                          |0.1.0      
entrypoints                      |0.3        
envirophat                       |1.0.0      
ExplorerHAT                      |0.4.2      
Flask                            |1.0.2      
fourletterphat                   |0.1.0      
frozenlist                       |1.2.0      
future                           |0.18.2     
gast                             |0.2.2      
google-auth                      |1.35.0     
google-auth-oauthlib             |0.4.6      
google-pasta                     |0.2.0      
gpiozero                         |1.6.2      
grpcio                           |1.43.0     
h5py                             |3.6.0      
html5lib                         |1.0.1      
idna                             |2.6        
importlib-metadata               |4.10.0     
isort                            |4.3.4      
itsdangerous                     |0.24       
jedi                             |0.13.2     
Jinja2                           |2.10       
keras                            |2.7.0      
Keras-Applications               |1.0.8      
Keras-Preprocessing              |1.1.2      
keyring                          |17.1.1     
keyrings.alt                     |3.1.1      
lazy-object-proxy                |1.3.1      
line-bot-sdk                     |2.0.1      
logilab-common                   |1.4.2      
lxml                             |4.3.2      
Markdown                         |3.3.6      
MarkupSafe                       |1.1.0      
mccabe                           |0.6.1      
microdotphat                     |0.2.1      
mote                             |0.0.4      
motephat                         |0.0.3      
multidict                        |5.2.0      
mypy                             |0.670      
mypy-extensions                  |0.4.1      
numpy                            |1.16.2     
oauthlib                         |2.1.0      
olefile                          |0.46       
opencv-python                    |4.5.5.62   
opt-einsum                       |3.3.0      
pantilthat                       |0.0.7      
parso                            |0.3.1      
pexpect                          |4.6.0      
pgzero                           |1.2        
phatbeat                         |0.1.1      
pianohat                         |0.1.0      
picamera                         |1.13       
piglow                           |1.2.5      
pigpio                           |1.78       
Pillow                           |5.4.1      
pip                              |18.1       
protobuf                         |3.19.1     
psutil                           |5.5.1      
pyasn1                           |0.4.8      
pyasn1-modules                   |0.2.8      
pycairo                          |1.16.2     
pycrypto                         |2.6.1      
pycups                           |1.9.73     
pyftdi                           |0.53.3     
pygame                           |1.9.4.post1
Pygments                         |2.3.1      
PyGObject                        |3.30.4     
pyinotify                        |0.9.6      
PyJWT                            |1.7.0      
pylint                           |2.2.2      
pyOpenSSL                        |19.0.0     
pyserial                         |3.4        
pysmbc                           |1.0.15.6   
python-apt                       |1.8.4.3    
pyusb                            |1.2.1      
pyxdg                            |0.25       
rainbowhat                       |0.1.0      
reportlab                        |3.5.13     
requests                         |2.21.0     
requests-oauthlib                |1.0.0      
responses                        |0.9.0      
roman                            |2.0.0      
rpi-ws281x                       |4.3.1      
RPi.GPIO                         |0.7.0      
rsa                              |4.8        
RTIMULib                         |7.2.1      
scrollphat                       |0.0.7      
scrollphathd                     |1.2.1      
SecretStorage                    |2.3.1      
Send2Trash                       |1.5.0      
sense-hat                        |2.2.0      
setuptools                       |60.2.0     
simplejson                       |3.16.0     
six                              |1.12.0     
skywriter                        |0.0.7      
sn3218                           |1.2.7      
soupsieve                        |1.8        
spidev                           |3.4        
ssh-import-id                    |5.7        
sysv-ipc                         |1.1.0      
tensorboard                      |2.0.2      
tensorflow                       |2.0.0      
tensorflow-estimator             |2.0.1      
termcolor                        |1.1.0      
thonny                           |3.3.6      
touchphat                        |0.0.1      
twython                          |3.7.0      
typed-ast                        |1.3.1      
typing-extensions                |4.0.1      
unicornhathd                     |0.0.4      
urllib3                          |1.24.1     
webencodings                     |0.5.1      
Werkzeug                         |0.14.1     
wheel                            |0.32.3     
wrapt                            |1.13.3     
yarl                             |1.7.2      
zipp                             |3.7.0

## 2. Basic Settings for Rasberry Pi

### Interfaces Configuration

* Enableed
    * Camera
    * SSH
    * VNC
    * I2C

![](https://i.imgur.com/ztFqUvE.jpg)

### Enabled I2C under CMD


> Before using the Adafruit PCA9685 16-Channel Servo Driver, you should enable the I2C, or the error will happen while running the code.

1. `sudo raspi-config`

![](https://i.imgur.com/8AVD5pw.png)

2. Enter Interface Options

![](https://i.imgur.com/ri29Ox7.png)

3. Select I2C

![](https://i.imgur.com/0Oc1FEc.png)

4. Enable I2C

![](https://i.imgur.com/MBKWtr3.png)

5. Enabled

![](https://i.imgur.com/754m44x.png)

6. Go back and finish

![](https://i.imgur.com/27nlGXk.png)


## 3. The Circuit Diagram

![](https://i.imgur.com/FfapWGr.jpg)

![](https://i.imgur.com/waKiiMN.jpg)

### Reference

https://github.com/adafruit/Fritzing-Library/blob/master/parts/retired/PCA9685%2016x12-bit%20PWM%20Breakout.fzpz

## 4. Line Bot Source Code

### Code

```python=
import picamera
from servoMotion import *
import os, shutil
from flask import Flask, request, abort
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import *
import json

from datetime import datetime
app = Flask(__name__)
# LINE BOT info
line_bot_api = LineBotApi('YOUR_CHANNEL_ACCESS_TOKEN')
handler = WebhookHandler('YOUR_CHANNEL_SECRET')
camera = picamera.PiCamera()

#object to control servo motor
servoControl = servoMotion(1,1)

folder = 'imagepreview'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    print(body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'


contents= json.load(open('BasicTable.json','r',encoding='utf-8'))
print(contents)

flex_message = FlexSendMessage(
    alt_text='hello',
    contents= json.load(open('BasicTable.json','r',encoding='utf-8'))
)
# Message event
@handler.add(MessageEvent)
def handle_message(event):
    message_type = event.message.type
    user_id = event.source.user_id
    reply_token = event.reply_token
    message = str(event.message.text)
    if(message == 'right'):
        FlexMessage = json.load(open('BasicTable.json','r',encoding='utf-8'))
        line_bot_api.reply_message(reply_token, FlexSendMessage('test1',FlexMessage))
        servoControl.turnRight()
    elif(message == 'left'):
        FlexMessage = json.load(open('returnPhotoTable.json','r',encoding='utf-8'))
        line_bot_api.reply_message(reply_token, FlexSendMessage('test2',FlexMessage))
        servoControl.turnLeft()
    elif(message == 'reset'):
        FlexMessage = json.load(open('returnPhotoTable.json','r',encoding='utf-8'))
        line_bot_api.reply_message(reply_token, FlexSendMessage('test2',FlexMessage))
        servoControl.resetStatus()
    else:
        line_bot_api.reply_message(reply_token, TextSendMessage(text=message))

arr = [0]

#handle postbackdata
@handler.add(PostbackEvent)
def handle_postback(event):
    postback_data = event.postback.data
    user_id = event.source.user_id
    reply_token = event.reply_token
    if postback_data == "getCurrentPhoto" or postback_data == "takePhoto":
        DIR = 'imagepreview/fit'+str(len(arr))+'.jpg'
        camera.capture(DIR)
        print('camera take the picture')
        FlexMessage = json.load(open('returnPhotoTable.json','r',encoding='utf-8'))
        url = "https://b1e3-140-115-214-31.ngrok.io/fit"+str(len(arr))+'.jpg'
        print(url)
        FlexMessage["hero"]["url"] = url
        FlexMessage["body"]["contents"][1]["contents"][2]["contents"][1]["text"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line_bot_api.reply_message(reply_token, FlexSendMessage('即時照片查看',FlexMessage))
        arr.append(0)
    elif postback_data == "resetStatus":
        line_bot_api.reply_message(reply_token, TextSendMessage(text="重置狀態中..."))
        servoControl.resetStatus()
    elif postback_data == "backToFunctionTable":
        FlexMessage = json.load(open('BasicTable.json','r',encoding='utf-8'))
        line_bot_api.reply_message(reply_token, FlexSendMessage('綜合功能表',FlexMessage))
    elif postback_data == "changeAngle":
        FlexMessage = json.load(open('changeRotation.json','r',encoding='utf-8'))
        line_bot_api.reply_message(reply_token, FlexSendMessage('鏡頭角度調整',FlexMessage))
    elif postback_data == "turnLeft":
        servoControl.turnLeft()
        if servoControl.getLeft()==0:
            line_bot_api.reply_message(reply_token, TextSendMessage(text="不能再向左轉囉~"))
        FlexMessage = json.load(open('changeRotation.json','r',encoding='utf-8'))
        line_bot_api.reply_message(reply_token, FlexSendMessage('鏡頭角度調整',FlexMessage))
    elif postback_data == "turnRight":
        servoControl.turnRight()
        if servoControl.getRight()==0:
            line_bot_api.reply_message(reply_token, TextSendMessage(text="不能再向右轉囉~"))
        FlexMessage = json.load(open('changeRotation.json','r',encoding='utf-8'))
        line_bot_api.reply_message(reply_token, FlexSendMessage('鏡頭角度調整',FlexMessage))
    '''elif postback_data == "takePhoto":
        FlexMessage = json.load(open('returnPhotoTable.json','r',encoding='utf-8')) 
        FlexMessage["body"]["contents"][1]["contents"][2]["contents"][1]["text"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line_bot_api.reply_message(reply_token, FlexSendMessage('即時照片查看',FlexMessage))'''
    

import os
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 80))
    app.run(host='0.0.0.0', port=port)
```

### Result

**1. postback_data == "getCurrentPhoto" or postback_data == "takePhoto":**

![](https://i.imgur.com/G7OqZfE.jpg)

**2. postback_data == "resetStatus":**

![](https://i.imgur.com/g5ICTRi.jpg)

**3. postback_data == "backToFunctionTable":**

![](https://i.imgur.com/k6uyqfV.jpg)

**4. postback_data == "changeAngle":**

![](https://i.imgur.com/bW3NUA5.jpg)

### Reference

https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder

https://stackoverflow.com/questions/10851906/python-3-unboundlocalerror-local-variable-referenced-before-assignment

https://developers.line.biz/flex-simulator/

## 5. Servo Motor Control

### Source Code

```python=
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
```

### Reference

https://learn.adafruit.com/adafruit-16-channel-servo-driver-with-raspberry-pi/using-the-adafruit-library

## 6. Camera Control

> This section is just for testing the camera can work properly with the picamera package.

### Source Code

```python=
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
```

### Reference

https://picamera.readthedocs.io/en/release-1.13/

## 7. Start two localhost Server

### Ngrok

Use ngrok to make the image captured by the camera visible on the outer web browser.

![](https://i.imgur.com/HJtEKKj.jpg)

Next, we can map the directory to the HTTPS URL and the directory can be accessed from outside with a web browser.  

![](https://i.imgur.com/pySosAe.jpg)

Access the image on a web browser with the URL.

![](https://i.imgur.com/9clsEwh.jpg)

Don't forget to change the URL in the python file every time you restart the ngrok.

![](https://i.imgur.com/BWLlUbo.jpg)

After changing the URL, you can execute the python file again.

![](https://i.imgur.com/yp4OUhr.jpg)

### SocketXP

We will use SocketXP to take the place of ngrok and get the webhook URL. Remember that we assign the port number to be 80 in the python file, so we should connect to port 80 with SocketXP.

![](https://i.imgur.com/UJh4Os9.jpg)

Next, we get the public URL that SocketXP issues for us.

![](https://i.imgur.com/KRhmcHh.jpg)

Fill the Webhook URL section with the URL we gained from SocketXP. Don't forget to add `/callback`.

![](https://i.imgur.com/ypIDYZk.jpg)


### Reference

https://www.socketxp.com/docs/guide/

https://ngrok.com/docs

## 8. Interact with Line Bot

If you finish the above steps successfully, then you can interact with the Line Bot happily.

If there is any problem you want to ask, you can issue me [here](https://github.com/QI-XIANG/IoT_Project).

![](https://i.imgur.com/G0axAcK.jpg)

Have fun and stay cool~

![](https://i.imgur.com/wGFSINE.jpg)

---

# Demo Video

## My Little Skynet Demo

### My Little Skynet Demo Part 1

https://www.youtube.com/watch?v=R9M28kE2mPs&ab_channel=%E4%B8%AD%E9%87%8E%E4%BA%8C%E4%B9%83

### My Little Skynet Demo Part 2

https://www.youtube.com/watch?v=Rzfvn844HQI&ab_channel=%E4%B8%AD%E9%87%8E%E4%BA%8C%E4%B9%83

---

# Appendix

## About the Appendix

In this appendix, I will introduce and teach you some key points for finishing the interesting IoT project. Don't be afraid when you encounter problems, because there are so many solutions buried in the Internet waiting for your visit. If you are overwhelmed while reading so much data search from the Internet, you can just take a rest and restart later. For me, I like to watch animation while taking a rest. 

> "Life is like a box of chocolates. You never know what you're gonna get."

![](https://i.imgur.com/wkomiWx.jpg)

## How to send flex message with FLEX MESSAGE SIMULATOR?

### FLEX MESSAGE SIMULATOR

You can use [FLEX MESSAGE SIMULATOR](https://developers.line.biz/flex-simulator/?status=success) to create your own flex message template. 

Let's follow the steps:

1. Click the link above, then go to the website. You may need to log in to your Line account before using it.
![](https://i.imgur.com/er10trE.png)
![](https://i.imgur.com/StaoTOI.png)
![](https://i.imgur.com/QnfMYxI.png)

2. Click on the component you want to configure. If you are familiar with the CSS flexbox, it may be quilt easier for you to use it.  
![](https://i.imgur.com/f4zkUEC.png)

3. After finishing the design of the flex message template, you can click "View as JSON" and store it for future use.
![](https://i.imgur.com/Jkqei7h.png)

### Reference

https://developers.line.biz/en/docs/messaging-api/using-flex-messages/

https://developers.line.biz/flex-simulator/

## How to send local image with Line Bot?

### Image Sending Restriction

While using Line Messaging API to send images, the image  URLs must use HTTPS over TLS 1.2 or later. If you don't follow the rule, the sending process won't be completed.

![](https://i.imgur.com/rCGykHA.png)


So, we need to use Ngrok to make the image we want to send follow the rule with HTTPS. 

![](https://i.imgur.com/RGH8b8o.png)

Let's follow two simple steps:

1. Change Directory to where the Ngrok file exists and run this in CMD:
```
./ngrok http "your image directory"
```
2. Result: Your local folder will be forwarded to the Internet

![](https://i.imgur.com/aPIqXEZ.png)
( **localhost:80** will change to your folder directory )

### Reference

https://ngrok.com/docs

https://developers.line.biz/en/docs/messaging-api/message-types/#image-messages

## Where to get the required components?

### Physical Store

大洋國際電子材料有限公司
地址： 320桃園市中壢區中平路59號
電話： 03-4252593
<iframe src="https://www.google.com/maps/embed?pb=!1m14!1m8!1m3!1d14469.328177399053!2d121.224617!3d24.9548151!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x0%3A0x9e026ed864b010da!2z5aSn5rSL5ZyL6Zqb6Zu75a2Q5p2Q5paZ5pyJ6ZmQ5YWs5Y-4!5e0!3m2!1szh-TW!2stw!4v1641677400102!5m2!1szh-TW!2stw" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy"></iframe>

### Web Store

[TAIWANIOT](https://www.taiwaniot.com.tw/)

![](https://i.imgur.com/ioTe3Vn.png)

[AMAZON](https://www.amazon.com/)

![](https://i.imgur.com/eSQ369Y.png)

[SHOPEE](https://shopee.tw/)

![](https://i.imgur.com/oH0sdPO.png)

[RUTEN](https://www.ruten.com.tw/)

![](https://i.imgur.com/JpRICIo.png)

other...

### Reference

https://goo.gl/maps/evT8tWF7uzT12ZCJ8

https://www.taiwaniot.com.tw/

https://www.amazon.com/

https://shopee.tw/

https://www.ruten.com.tw/

## How to draw the circuit diagram?

### Tinkercad
[Tinkercad](https://www.tinkercad.com/) is a free, easy-to-use web app that equips the next generation of designers and engineers with the foundational skills for innovation: 3D design, electronics, and coding!

![](https://i.imgur.com/86Qbz6G.jpg)

One problem will happen when you use Tinkercad to draw circuits diagram. Tinkercad doesn't provide Raspberry Pi as its drawable component. So, if you want to place Raspberry Pi in your circuit diagram, you need to find a tool or platform that supports Raspberry Pi.

![](https://i.imgur.com/BkAeP38.png)

### Fritzing

[Fritzing](https://fritzing.org/) is an open-source hardware initiative that makes electronics accessible as a creative material for anyone. We offer a software tool, a community website and services in the spirit of Processing and Arduino, fostering a creative ecosystem that allows users to document their prototypes, share them with others, teach electronics in a classroom, and layout and manufacture professional PCBs.

![](https://i.imgur.com/u2z13Uv.png)

You can't download Fritzing Application on their website without pay. Then you can download it from other websites, but I can't tell you where to download it because this may cause legal problems.

The first time you open the application, the window you will see just like the following picture. Then search the Raspberry Pi components, and you could find so many kinds of them. We choose **Raspberry Pi B+** from them. 

![](https://i.imgur.com/NeIyJLi.png)

Click and drag **Raspberry Pi B+** to the main blank area, you can start to draw your circuit diagram smoothly.

![](https://i.imgur.com/Li8jlbC.png)

### Reference

https://www.tinkercad.com/

https://fritzing.org/

https://github.com/adafruit/Fritzing-Library

https://blog.cavedu.com/2013/09/21/%E5%9C%A8fritzing%E6%96%B0%E5%A2%9Eraspberry-pi%E6%A8%A1%E7%B5%84/

## Other useful tools while developing your own IoT project

### [Google Codelabs](https://colab.research.google.com/notebooks/welcome.ipynb?hl=en-US#scrollTo=5fCEDCU_qrC0)
![](https://i.imgur.com/6UJPE4q.png)

What is Colab?

Colab, or "Colaboratory", allows you to write and execute Python in your browser, with:

* Zero configuration required
* Free access to GPUs
* Easy sharing

Whether you're a student, a data scientist or an AI researcher, Colab can make your work easier. Watch Introduction to Colab to learn more, or just get started below!

While training the image classification model, you can use GPUs from google codelabs to accelerate the computation. 

### [Microsoft Visual Studio Code](https://code.visualstudio.com/)

Visual Studio Code is a lightweight but powerful source code editor which runs on your desktop and is available for Windows, macOS and Linux. It comes with built-in support for JavaScript, TypeScript and Node.js and has a rich ecosystem of extensions for other languages (such as C++, C#, Java, Python, PHP, Go) and runtimes (such as .NET and Unity).

![](https://i.imgur.com/4lg88E7.jpg)

The Raspbian OS built-in code editors like Geany or Thonny is not convenient while coding for a long time. You can download Microsoft Visual Studio Code as your code editor, then copy and paste the code to Raspberry Pi.

### [Jupyter Notebook](https://jupyter.org/)

The Jupyter Notebook is the original web application for creating and sharing computational documents. It offers a simple, streamlined, document-centric experience.

![](https://i.imgur.com/A1WyRkh.png)

The best characteristic of Jupyter Notebook is that it allows running the code section separately. This may be helpful while debugging

### [SocketXP](https://www.socketxp.com/)

What is SocketXP?

SocketXP is a cloud based IoT Device Management and Remote Access Platform that empowers you to remotely connect, login, configure, debug, upgrade, track, monitor and manage millions of IoT, IIoT devices or Raspberry Pi or any Linux machines installed in your customer's local network behind NAT router and firewall.

![](https://i.imgur.com/ixRb6Pe.png)

Note that we need a webhook to start our Line Bot, then you can use SocketXP as a webhook and take the place of ngrok.

### Reference

https://colab.research.google.com/notebooks/welcome.ipynb?hl=en-US#scrollTo=5fCEDCU_qrC0

https://code.visualstudio.com/

https://jupyter.org/

https://www.socketxp.com/

---
