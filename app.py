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
from flask import Flask,jsonify
from flask import Flask, render_template,url_for,request

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

@app.route('/image-rotation/<string:input_path>', methods = ['GET'])
def predict(input_path) :
    global model

    #input_img = load_img(input_path)
    input_img = cv2.imread(input_path)
    #X_test_np = np.array(input_img)
    resized_input = cv2.resize(input_img,  (256, 256), interpolation = cv2.INTER_NEAREST)
    resized_input = np.expand_dims(resized_input, axis = 0)
    #print("SHAPE : ", resized_input.shape)
    predicted_tensor = model.predict(resized_input)
    predicted_angle = argmax(predicted_tensor, axis = 1)
    print("predicted_angle : ", np.array(predicted_angle)[0])

    #out = model.evaluate_generator(
    '''RotNetDataGenerator(
        X_test_np,
        batch_size=128,
        preprocess_func=binarize_images,
        shuffle=True
    )''' 
    #steps=len(y_test) / batch_size

    return jsonify({"rotationAngle" : int(np.array(predicted_angle)[0])})





#Machine Learning code goes here
if __name__ == '__main__':
    app.run(debug=True)
    