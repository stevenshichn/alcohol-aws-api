# author: Steven Shi, 14 Jul 2018

import traceback
from flask import Flask, request, jsonify
from PIL import Image, ExifTags
from keras.models import model_from_json
import numpy as np
import tensorflow as tf
from scipy.misc import imresize
import boto3
import pickle

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
BUCKET_NAME = 'image-recognition-models'
model_file_path = 'alcohol_cnn.h5'
model_weight_file_path = 'model_vgg16.h5'
model_json_file_path = 'model_vgg16.json'

app = Flask(__name__)

s3 = boto3.client('s3', region_name='us-east-2')
boto3.resource('s3').Bucket(BUCKET_NAME).download_file(model_file_path, model_file_path)
boto3.resource('s3').Bucket(BUCKET_NAME).download_file(model_json_file_path, model_json_file_path)
data = None # to store json string from json file
with open(model_json_file_path, encoding='utf-8') as weight_file:
    data = weight_file.read()

graph = tf.get_default_graph()
model = model_from_json(data)
model.load_weights(model_weight_file_path)

@app.route("/predict", methods=["POST"])
def predict_image():
    #Preprocess the image so that it matches the training input
    image = request.form.get('image')
       
    image = Image.open(image)
    image = rotate_by_exif(image)
    resized_image = imresize(image, (224, 224)) / 255.0

    # Model input shape = (224,224,3)
    # [0:3] - Take only the first 3 RGB channels and drop ALPHA 4th channel in case this is a PNG
    pred = ml_predict(resized_image[:, :, 0:3])
    print('result' + str(pred))
    # Prepare and send the response.
    print(pred.take(0, axis=0))
    
    prediction = {'result' : np.float64(pred.take(0, axis=0)[0])}
    return jsonify(prediction)

def ml_predict(image):
    with graph.as_default():
        # Add a dimension for the batch
        prediction = model.predict(image[None, :, :, :])
    return prediction

def rotate_by_exif(image):
    try :
        for orientation in ExifTags.TAGS.keys() :
            if ExifTags.TAGS[orientation]=='Orientation' : break
        exif=dict(image._getexif().items())
        if not orientation in exif:
            return image

        if   exif[orientation] == 3 :
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6 :
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8 :
            image=image.rotate(90, expand=True)
        return image
    except:
        traceback.print_exc()
        return image

   
if __name__ == "__main__":
    app.run(debug = True)