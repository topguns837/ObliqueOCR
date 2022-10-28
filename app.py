import flask
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import load_img
import tensorflow_datasets as tfds
from tensorflow import argmax

from utils import RotNetDataGenerator, binarize_images

from PIL import Image
import cv2

from flask import Flask, render_template,url_for,request, jsonify

app = Flask(__name__)

def dataset_to_numpy(ds):
    """
    Convert tensorflow dataset to numpy arrays
    """
    images = []   

    # Iterate over a dataset
    for i, image  in enumerate(tfds.as_numpy(ds)):
        images.append(image)
        #print(np.array(image).shape)         
    return np.vstack(images) 


def angle_difference(x, y):
    
    return 180 - abs(abs(x - y) - 180)

def angle_error(y_true, y_pred):
    
    diff = angle_difference(K.argmax(y_true), K.argmax(y_pred))
    return K.mean(K.cast(K.abs(diff), K.floatx()))





model_location = 'rotnet_model.h5'
model = load_model(model_location, custom_objects={'angle_error': angle_error})

@app.route('/image-rotation', methods = ['GET','POST' ])
def predict() :
    global model
    input_path = ""
    
    if request.method == 'POST' :
        
        json = request.get_json()
        input_path = json['url']
         
        
        input_img = cv2.imread(input_path)
        
        resized_input = cv2.resize(input_img,  (256, 256), interpolation = cv2.INTER_NEAREST)
        resized_input = np.expand_dims(resized_input, axis = 0)
        
        predicted_tensor = model.predict(resized_input)
        predicted_angle = argmax(predicted_tensor, axis = 1)
               
        
        return jsonify({"rotationAngle" : int(np.array(predicted_angle)[0])})

    
    

if __name__ == '__main__':
    app.run(debug=True)
    