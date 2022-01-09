from keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np


def Prediction(path):

    model = load_model('model.h5')
    model.load_weights('first_try.h5')

    model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

    img_width, img_height = 150, 150

    # predicting image
    img = image.load_img(path, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    kind = model.predict(images, batch_size=10).astype("int32")
    print(kind[0][0])