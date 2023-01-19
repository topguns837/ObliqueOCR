import numpy as np
import streamlit as st

import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import load_img
import tensorflow_datasets as tfds
from tensorflow import argmax

from utils import *

from PIL import Image
import cv2
import os

os.chdir(os.getcwd())

filename = "input.jpg"

model = load_model("classification_efficientnet_augmented.h5", custom_objects = {"angle_error_classification" : angle_error_classification})

def save_uploadedfile(uploadedfile):
    global filename
    #filename = uploadedfile.name

    with open(os.path.join("temp","input.jpg"),"wb") as f:
        f.write(uploadedfile.getbuffer())    
    

    return st.success("Successfuly uploaded file ")



st.write("Topic")

image_file = st.file_uploader("Enter the input image : ", type = ['png','jpeg','jpg'])


if image_file is not None : 
    save_uploadedfile(image_file)

    print("Current Dir : ", os.getcwd())

    input = cv2.imread(os.path.join(os.getcwd() + "/temp/input.jpg")) 
    st.image(input, caption = "Input Image", width = 200)    

    if input is None :
        print("Image is None")

    else :
        img_input = cv2.resize(input, (224, 224))
        img_input = np.expand_dims(img_input, axis = 0) 
        
        pred_angle = np.argmax(model.predict(img_input)) 

        st.write("Predicted Angle : ", pred_angle)

        rotated_img = rotate(input, -pred_angle)
        
        st.image(rotated_img, caption = "Rotated Image", width = 200)

        roi = get_roi(rotated_img)

        st.image(roi, caption = "ROI", width = 200)

        text = ocr(roi)

        st.write("Extracted Text : ", text[0])



