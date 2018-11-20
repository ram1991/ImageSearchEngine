from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
import cv2
import h5py
import os
import json
import datetime
import time


model = ResNet50(weights = 'imagenet', include_top = False, pooling = 'avg')

image_path = 'image.jpg'

image = load_img(image_path, target_size = (224,224))
image = image.img_to_array(image)
image = np.expand_dims(image, axis = 0)
image = preprocess_input(image)

predict = model.predict(image)
