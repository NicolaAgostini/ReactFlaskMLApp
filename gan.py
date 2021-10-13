"""TRY ON FACE DATASET

"""

import tensorflow as tf
import os
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pathlib 
import PIL
from tensorflow.keras import layers
import time

from tensorflow import keras
from tensorflow.keras.utils import Progbar
import tensorflow.keras as keras

import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


train_data_path ="./img_align_celeba_png/"



#len(glob.glob(os.path.join(train_data_path,"*")))

train_data_path = pathlib.Path(train_data_path)
print(train_data_path)

dataset = tf.data.Dataset.list_files(str(train_data_path/'*.png'), shuffle=True) #load a dataset from a folder

#if fails try to load with keras api
"""
data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale= 1./255)
training_data = data_gen.flow_from_directory(directory = train_data_path,
                                             target_size= (224,224),
                                             batch_size = 64,
                                             shuffle = True,
                                             class_mode = None,
                                             classes = None
                                            )

# if fails with keras api too

path_celeb = []
train_path_celeb = "/content/drive/My Drive/faces/img_align_celeba/img_align_celeba"
for path in os.listdir(train_path_celeb):
    if '.jpg' in path:
        path_celeb.append(os.path.join(train_path_celeb, path))

new_path=path_celeb[0:50000] # take only 50k


crop = (30, 55, 150, 175) #croping size for the image so that only the face at centre is obtained
images = [np.array((Image.open(path).crop(crop)).resize((64,64))) for path in new_path]

for i in range(len(images)):
    images[i] = ((images[i] - images[i].min())/(255 - images[i].min())) # if activation is sigmoid
    #images[i] = images[i]*2-1  #uncomment this if activation is tanh for generator last layer
 
images = np.array(images) 

ds_train=images

print(ds_train.shape)

ds_train_paths = ds_train_paths.take(train_size)

for f in dataset.take(5):
  print(f.numpy())
"""
print(len(list(dataset))) # total number of images in training set

def decode_img(file_path):
    file = tf.io.read_file(file_path)
    img = tf.image.decode_png(file, channels=3)
    img = tf.image.resize(img , [224,224])
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img / 255.0
    return img

BUFFER_SIZE = 60000
BATCH_SIZE = 64
shuffle_buffer = 1000
epoches = 100
#BIG dataset
#ds_train = dataset.map(decode_img).cache().shuffle(shuffle_buffer).batch(BATCH_SIZE, drop_remainder=True)
#SMALL dataset
ds_train = dataset.take(30000).map(decode_img).cache().shuffle(shuffle_buffer).batch(BATCH_SIZE, drop_remainder=True)
print(len(ds_train))
def get_noise(batch_size , z_dim):
    return tf.random.normal([batch_size,z_dim])

a = get_noise(BATCH_SIZE, 150)


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*512, use_bias=False, input_shape=(150,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 512)))
    assert model.output_shape == (None, 7, 7, 512) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 28, 28, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 56, 56, 64)
    
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 112, 112, 32)

    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid'))
    assert model.output_shape == (None, 224, 224, 3)

    return model

#testing generator
s = tf.random.normal([1,150])

gen = make_generator_model()
x = gen(s)
assert x.shape == (1,224,224,3)

def make_discriminator_model():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(32,(5,5), strides = (2,2) , padding = "same"  ,input_shape = (224,224,3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(64,(5,5), strides = (2,2) , padding = "same"  ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128,(5,5), strides = (2,2) , padding = "same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Conv2D(256,(5,5), strides = (2,2) , padding = "same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

#testing Discriminator
crit = make_discriminator_model()
s = tf.random.normal([1,224,224,3])
x = crit(s)


generator = make_generator_model()

noise = tf.random.normal([1, 150])
generated_image = generator(noise, training=False)

print("generating random image")
plt.imshow(generated_image[0, :, :, :])
plt.savefig("./ganOutput/"+"RandomExample.png")

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print("discrimitor output on previous random image")
print (decision)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)

def generator_loss(dx_of_gx):
    # Labels are true here because generator thinks he produces real images. 
    return cross_entropy(tf.ones_like(dx_of_gx), dx_of_gx)

def discriminator_loss(d_x, g_z, smoothing_factor = 0.98): #smoothing factor to allow some errors on discrimitator
    """
    d_x = real output
    g_z = fake output
    """
    real_loss = cross_entropy(tf.ones_like(d_x) * smoothing_factor, d_x) # If we feed the discriminator with real images, we assume they all are the right pictures --> Because of that label == 1
    fake_loss = cross_entropy(tf.zeros_like(g_z), g_z) # Each noise we feed in are fakes image --> Because of that labels are 0. 
    total_loss = real_loss + fake_loss
    
    return total_loss

generator_optimizer = tf.keras.optimizers.Adam(0.002,0.5,0.99) 
discriminator_optimizer = tf.keras.optimizers.Adam(0.002,0.5,0.99)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 150])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def train(dataset, epochs):
  
  for epoch in range(epochs):
    progress_bar = Progbar(30000)
    gen_loss_list = []
    disc_loss_list = []

    for image_batch in dataset:
      gen_loss, disc_loss = train_step(image_batch)

      gen_loss_list.append(gen_loss)
      disc_loss_list.append(disc_loss)
      
      progress_bar.add(64, [('Generator loss',gen_loss), ('Discriminator loss', disc_loss)]  ) #batch

    mean_g_loss = sum(gen_loss_list)/len(gen_loss_list)
    mean_d_loss = sum(disc_loss_list)/len(disc_loss_list)
    gen_loss_list.append(mean_g_loss)
    disc_loss_list.append(mean_d_loss)

    print (f'Epoch {epoch+1}, gen loss={mean_g_loss},critic loss={mean_d_loss}')

    # Save the model every 5 epochs
    #if (epoch + 1) % 5 == 0:
    
    
    seed = tf.random.normal([1, 150])
    fake_image = tf.reshape(generator(seed), shape = (224,224,3))
    print("{}/{} epoches".format(epoch, epoches))
    #plt.imshow(fake_image, cmap = "gray")

    plt.imshow(fake_image.numpy())
    plt.savefig("./ganOutput/"+str(epoch)+".png")
    #print(fake_image)

generator = make_generator_model()
discriminator = make_discriminator_model()

generator_json = generator.to_json()
with open("GANgenerator.json", "w") as json_file:
    json_file.write(generator_json)


train(ds_train,epoches)

# serialize weights to HDF5
generator.save_weights("GANgenerator.h5")

print("Saved model to disk")

#tf.saved_model.save(generator , "/content/drive/My Drive/faces/training_checkpoints/")
#generator.save('./Gan_tf100',save_format='tf')
#loaded_model = tf.keras.models.load_model('/content/drive/My Drive/faces/training_checkpoints/MyModel_tf')
#generator.save('/content/drive/My Drive/faces/training_checkpoints/')
"""
import os

checkpoint_dir = '/content/drive/My Drive/faces/training_checkpoints/'
checkpoint_prefix = '/content/drive/My Drive/faces/training_checkpoints/'
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

ckpt = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
manager = tf.train.CheckpointManager(ckpt, '/content/drive/My Drive/faces/training_checkpoints/', max_to_keep=3)

while True:pass

trained_generator = tf.keras.models.load_model('/content/drive/My Drive/faces/training_checkpoints/MyModel_tf200')

# Check its architecture
trained_generator.summary()

seed = tf.random.normal([1, 100])
fake_image = tf.reshape(trained_generator(seed), shape = (56,56,3))

plt.imshow(fake_image.numpy())
plt.show()

"""