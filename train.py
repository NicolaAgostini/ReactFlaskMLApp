

import keras
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense
from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

from keras.models import model_from_json

import cv2
import numpy as np


# Number of epochs
epochs = 50

# Batch size
batch_size = 16

# Number of categories
classes = 1

# Image size
image_size = (<your size>,<your size>)

# Number of training and validation samples
number_of_images_training = <your number of training images>
number_of_images_validation = <your number of validation images>

# Steps per epoch and validation steps per epoch
steps_per_epoch = number_of_images // batch_size
validation_steps = number_of_images // batch_size

# Augment the data during training
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
rescale=1./255)

validation_datagen = ImageDataGenerator(
    rescale=1./255)
    
base_model= VGG16(weights='imagenet', include_top=False, input_shape=(image_size,3))


top_model = Flatten()(base_model.output)
top_model = Dense(256, activation='relu')(top_model)
top_model = Dense(classes, activation='softmax')(top_model)



model = Model(inputs=base_model.input, outputs=top_model)



# 'binary_crossentropy' for 2 classes, 'categorical_crossentropy' > 2 classes
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

for layer in model.layers[:20]:
    layer.trainable = False
    
model.summary()

train_generator = train_datagen.flow_from_directory(
        # Add training directory
        'data/train',
        target_size=(150, 150),
        batch_size=batch_size,
    
        # 'binary' for 2 classes, 'categorical' > 2 classes
        class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
        # Add validation directory
        'data/validation',
        target_size=(150, 150),
        batch_size=batch_size,
    
        # 'binary' for 2 classes, 'categorical' > 2 classes
        class_mode='binary')
        

label_map = (train_generator.class_indices)

# Invert the dictionary
inverted_label_map = {v:k for k,v in label_map.items()}
print(inverted_label_map)


model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps)
        
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
model.save_weights('your_model.h5')
