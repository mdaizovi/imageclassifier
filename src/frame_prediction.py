import cv2
import numpy as np
from pprint import pprint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

IMG_TARGET_SIZE = (224, 224)


def predict_frame(frame):
    """
    Takes a [224x224] frame from your webcam and calculates a prediction with a pre-trained network.
    """

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    a = image.img_to_array(img)
    a = preprocess_input(a)

    # reshape image to (1, 224, 224, 3)
    a = a.reshape(1, 224, 224, 3)

    # apply pre-processing
    img = preprocess_input(img)

    m = MobileNetV2(weights='imagenet', include_top=True)
    m.summary()
    p = m.predict(a)

    pprint(decode_predictions(p, 10))

    return p
