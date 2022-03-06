2021 IoT Final Project

I will update more content here soon!

# My Little Skynet 

###### tags: `IoT` `My Little Skynet` `Project Doc Github Ver`

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
        1. Raspberry Pi will keep getting images from the connected camera.
        2. Raspberry Pi will use the image and recognition model to identify if there exists any stranger. 
        3. After identifying by the Raspberry Pi there does not exist any stranger. 
        4. Only one thing will happen:
            *  The light‑emitting diode will keep lighting with green light.
        5. User can interact with Line Bot and press down the "taking photo" button to ask Raspberry Pi for taking pictures and return them.
        6. Raspberry Pi will use Line to send the image gotten from the connected camera every 5 seconds for 1 minute with 12 images in total. 

---

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
| Buzzer  | 1 |  
| Intel® Neural Compute Stick 2 | 1 |
| Adhesive Tape  | 1 |
| Carton| 1 |
|PCA9685: 16-Channel, 12-Bit PWM Fm+ I²C-Bus LED Controller| 1 |
|GPIO Expansion T Board for Raspberry Pi|1|
|Rainbow Cable|1|
|Adafruit PCA9685 16-Channel Servo Driver|1|

---

## :books: Line Bot

###### tags: `IoT` `My Little Skynet`

>First of all, you should have a Line account before creating your Line Bot. 

### 1. [Line Developers](https://developers.line.biz/en/)

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
 
---

### 2. [LINE Messaging API SDK for Python](https://github.com/line/line-bot-sdk-python)

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

---

### 3. Webhook

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



---

### 4. [Ngrok](https://ngrok.com)

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
        
---

### 5. Reference

https://medium.com/@justinlee_78563/line-bot-%E7%B3%BB%E5%88%97%E6%96%87-%E4%BB%80%E9%BA%BC%E6%98%AF-webhook-d0ab0bb192be

https://ngrok.com/product

https://developers.line.biz/en/docs/messaging-api/line-bot-sdk/#messaging-api-sdks

https://github.com/line/line-bot-sdk-python

https://developers.line.biz/en/

https://www.webfx.com/tools/emoji-cheat-sheet/

---

## :sailboat: CNN Model

###### tags: `IoT` `My Little Skynet`

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

---

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
