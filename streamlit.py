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

flip_flag = False

filename = "input.jpg"

col1, col2, col3 = st.columns(3)

model = load_model("model/classification_efficientnet_augmented.h5", custom_objects = {"angle_error_classification" : angle_error_classification})

def save_uploadedfile(uploadedfile):
    global filename

    with open(os.path.join("temp","input.jpg"),"wb") as f:
        f.write(uploadedfile.getbuffer())       


st.markdown("<h1 style='text-align: center; color: grey;'>ObliqueOCR</h1>", unsafe_allow_html=True)

st.write("\n\n")


image_file = st.file_uploader("Enter the input image : ", type = ['png','jpeg','jpg'])


if image_file is not None : 
    save_uploadedfile(image_file)

    input = cv2.imread(os.path.join(os.getcwd() + "/temp/input.jpg")) 

    st.image(input, caption = "Input Image", width = 200)  
    st.write("\n\n")

    if input is None :
        print("Image is None")

    else :
        img_input = cv2.resize(input, (224, 224))
        img_input = np.expand_dims(img_input, axis = 0) 
        
        pred_angle = np.argmax(model.predict(img_input)) 

        st.success("Predicted Angle : {}°".format(pred_angle))
        st.write("\n\n")

        rotated_img = rotate(input, -pred_angle)
        
        st.image(rotated_img, caption = "Rotated Image", width = 200)
        st.write("\n\n")

        roi = get_roi(rotated_img)        

        flip_flag = st.button("Rotate ROI")

        if flip_flag : 
            roi = rotate(roi, 180)
 
        st.image(roi, caption = "ROI", width = 200)
        st.write("\n\n")        

        text = ocr(roi)

        if text is not [""] :
            st.success("Extracted Text : " + text[0])
        else :
            st.error("Not able to extract text")



