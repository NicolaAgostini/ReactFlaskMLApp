from flask import Flask, request, send_from_directory
import requests

from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json

import keras
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense
from keras.models import Model
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
from keras.models import model_from_json

import cv2

import tensorflow as tf



# Load network architecture
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load weights into a new model
loaded_model.load_weights("model.h5")

# Compile the loaded model
opt = SGD(lr=0.001, momentum=0.9)
loaded_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])



#app = Flask(__name__)
#app = Flask(__name__, static_folder='frontend/build', static_url_path='/')
app = Flask(__name__, static_folder=os.path.abspath("frontend/build"), static_url_path='/')
# Allow 
CORS(app)

# Allowed file extransions
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return app.send_static_file('index.html')


@app.errorhandler(404)   
def not_found(e):   
  return app.send_static_file('index.html')

"""
@app.route("/")
def hello():
	return app.send_static_file('index.html')
"""
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':

		# Check if the post request has the file part
		if 'file' not in request.files:
			return "No file part"
		file = request.files['file']

		if file and allowed_file(file.filename):

			# Get image from frontend stream
			image_from_frontend = Image.open(request.files['file'].stream)
			# Send uploaded image for prediction
			predicted_image_class = predict_img(image_from_frontend)

	# Send predicted class back to frontend
	return json.dumps(predicted_image_class)

def predict_img(image_from_frontend):
	
	# Choose same image size as the model is trained on
	image_size = (224,224)

	# Add your available classes for predicition
	classes = {0:'cat', 1:'dog'}

	# Transform image from bytes to numpy array
	img_to_np_array = np.array(image_from_frontend)

	# Resize image to the same image as training image size
	res = cv2.resize(img_to_np_array, dsize=image_size, interpolation=cv2.INTER_CUBIC)

	# Reshape (image_size,3) to (1,image_size,3)
	img_for_prediction = res[np.newaxis,...]

	# Predict image class
	prediction = loaded_model.predict(img_for_prediction)

	# Get image prediction class
	#prediction_class = prediction.argmax(axis=-1)

	# Get predicted class
	#prediction_label = classes[prediction_class[0]]

	prediction_label = classes[int(prediction[0])]	
	
	return prediction_label



#RUN with Flask run on the folder 

"""	
if __name__ == '__main__':
      app.run(host='0.0.0.0', port=80)

"""