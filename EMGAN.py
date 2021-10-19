import tensorflow as tf
from keras import backend, layers
from numpy import expand_dims
from numpy import mean
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal
from keras.constraints import Constraint
from matplotlib import pyplot
import numpy as np
from matplotlib import pyplot as plt
import os
from keras.datasets.mnist import load_data
from tqdm import tqdm
from keras.models import model_from_json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true*y_pred)

# clip model weights 
class ClipConstraint(Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value
 
	# clip model weights when called from keras training
	def __call__(self, weights):
		return backend.clip(weights, -self.clip_value, self.clip_value)
 
	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}




def critic_model(in_shape=(128,128,3)):  # 3 if rgb
	
    init = RandomNormal(stddev=0.02)
	# weight constraint
    const = ClipConstraint(0.01)
    model = Sequential()

    model.add(tf.keras.layers.Conv2D(32,(3,3), strides = (2,2) , padding = "same"  ,input_shape = in_shape, kernel_constraint=const, kernel_initializer=init))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(64,(3,3), strides = (2,2) , padding = "same",kernel_initializer=init, kernel_constraint=const))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(64,(3,3), strides = (2,2) , padding = "same",kernel_initializer=init, kernel_constraint=const))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    
    

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="linear"))

    
    return model
    """
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# weight constraint
	const = ClipConstraint(0.01)
	# define model
	model = Sequential()
	# downsample to 14x14
	model.add(Conv2D(8, (16,16), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const, input_shape=in_shape))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))
	# downsample to 7x7
	model.add(Conv2D(16, (8,8), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))
	# scoring, linear activation
	model.add(Conv2D(32, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))
	# scoring, linear activation
	model.add(Flatten())
	model.add(Dense(1))
	# compile model
	
	return model
    """
def generator_model(latent_dim):
	
	init = RandomNormal(stddev=0.02)
	model = Sequential()

	model.add(layers.Dense(16*16*64, use_bias=False, input_shape=(latent_dim,)))
	model.add(layers.LeakyReLU())
	model.add(layers.Reshape((16, 16, 64)))

	model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init))
	assert model.output_shape == (None, 32, 32, 64)
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init))
	assert model.output_shape == (None, 64, 64, 32)

	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init))
	assert model.output_shape == (None, 128, 128, 16)

	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(Conv2D(16, (3,3), padding='same', kernel_initializer=init)) # 3 if rgb
	model.add(layers.LeakyReLU())
	model.add(Conv2D(3, (3,3), activation='tanh', padding='same', kernel_initializer=init)) # 3 if rgb
	assert model.output_shape == (None, 128, 128, 3)


	return model
	"""
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# define model
	model = Sequential()
	# foundation for 7x7 image
	n_nodes = 128 * 16 * 16
	model.add(Dense(n_nodes, kernel_initializer=init, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((16, 16, 128)))
	# upsample to 16x16
	model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 32x32
	model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))
	# output 64x64x1
	model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))
	# output 128x128x1
	model.add(Conv2D(3, (4,4), activation='tanh', padding='same', kernel_initializer=init))
	assert model.output_shape == (None, 128, 128, 3)
	return model
    """
# define the combined generator and critic model, for updating the GENERATOR (GAN)
def define_gan(generator, critic):
	# make weights in the critic not trainable
	for layer in critic.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(generator)
	# add the critic
	model.add(critic)
	# compile model
	
	return model


def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# select images
	X = dataset[ix]
	# generate class labels, -1 for 'real'
	y = -ones((n_samples, 1))
	return X, y



# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input


def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = generator.predict(x_input)
	# create class labels with 1.0 for 'fake'
	y = ones((n_samples, 1))
	return X, y

# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist):
	# plot history
	pyplot.plot(d1_hist, label='crit_real')
	pyplot.plot(d2_hist, label='crit_fake')
	pyplot.plot(g_hist, label='gen')
	pyplot.legend()
	pyplot.savefig('EMGan_plot_line_plot_loss.png')
	pyplot.close()
	
# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, latent_dim, n_samples=100):
	# prepare fake examples
	X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
	# scale from [-1,1] to [0,1]
	X = (X + 1) / 2.0
	# plot images
	for i in range(10 * 10):
		# define subplot
		pyplot.subplot(10, 10, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(X[i, :, :, :])
	# save plot to file
	filename1 = 'ECGanOutput/generated_plot_%04d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'EMGanModel_%04d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))


# train the generator and critic
def train(g_model, c_model, gan_model, dataset, latent_dim, n_epochs=30, n_batch=128, n_critic=8):
	# calculate the number of batches per training epoch
	bat_per_epo = int(dataset.shape[0] / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	progbar = tqdm(range(n_steps))
	# calculate the size of half a batch of samples
	half_batch = int(n_batch / 2)
	# lists for keeping track of loss
	c1_hist, c2_hist, g_hist = list(), list(), list()
	# manually enumerate epochs
	for i in progbar:
		# update the critic more than the generator, usually 5 times
		c1_tmp, c2_tmp = list(), list()
		for _ in range(n_critic):
			# get randomly selected 'real' samples
			X_real, y_real = generate_real_samples(dataset, half_batch)
			# update critic model weights
			c_loss1 = c_model.train_on_batch(X_real, y_real)
			c1_tmp.append(c_loss1)
			# generate 'fake' examples
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# update critic model weights
			c_loss2 = c_model.train_on_batch(X_fake, y_fake)
			c2_tmp.append(c_loss2)
		# store critic loss
		c1_hist.append(mean(c1_tmp))
		c2_hist.append(mean(c2_tmp))
		# prepare points in latent space as input for the generator
		X_gan = generate_latent_points(latent_dim, n_batch)
		# create inverted labels for the fake samples
		y_gan = -ones((n_batch, 1))
		# update the generator via the critic's error
		g_loss = gan_model.train_on_batch(X_gan, y_gan)
		g_hist.append(g_loss)
		# summarize loss on this batch
		progbar.set_description('>%d, c1=%.3f, c2=%.3f g=%.3f' % (i+1, c1_hist[-1], c2_hist[-1], g_loss))
		# evaluate the model performance every 'epoch'
		if (i+1) % bat_per_epo == 0:
			summarize_performance(i, g_model, latent_dim)
	# line plots of loss
	plot_history(c1_hist, c2_hist, g_hist)


def load_real_samples(filename): #load img from format .npz
    # load dataset
    data = np.load(filename)
    # extract numpy array
    X = data['arr_0']
    # convert from ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    return X

# load images
def load_real_samples_MNIST():
	# load dataset
	(trainX, trainy), (_, _) = load_data()
	# select all of the examples for a given class
	selected_ix = trainy == 7
	X = trainX[selected_ix]
	# expand to 3d, e.g. add channels
	X = expand_dims(X, axis=-1)
	# convert from ints to floats
	X = X.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	return X

def startTraining():
	# size of the latent space
	latent_dim = 150
	# create the critic
	critic = critic_model()
	#parallel_critic=multi_gpu_model(critic,gpus=1)
	opt_critic = RMSprop(lr=0.00005)
	critic.compile(loss=wasserstein_loss, optimizer=opt_critic)
	# create the generator
	generator = generator_model(latent_dim)
	#parallel_generator=multi_gpu_model(generator,gpus=1)
	# create the gan
	print(critic.summary())
	print(generator.summary())
	gan_model = define_gan(generator, critic)
	#parallel_gan=multi_gpu_model(gan_model,gpus=1)
	opt_gan = RMSprop(lr=0.00005)
	gan_model.compile(loss=wasserstein_loss, optimizer=opt_gan)
	#generator_json = generator.to_json()
	#with open("GANgenerator.json", "w") as json_file:
	#json_file.write(generator_json)
	dataset = load_real_samples("img_align_celeba_128.npz")
	#dataset = load_real_samples_MNIST()
	print("Loaded: ",dataset.shape)
	#plt.imshow(dataset[3])
	#plt.show()
	# train model
	train(generator, critic, gan_model, dataset, latent_dim)



def main():
	startTraining()
	"""
	json_file = open('GANgenerator.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	# Load weights into a new model
	loaded_model.load_weights("EMGanModel_11700.h5")

	# Compile the loaded model
	opt = RMSprop(lr=0.00005)
	loaded_model.compile(loss=wasserstein_loss, optimizer=opt)
	X, _ = generate_fake_samples(loaded_model, 150, 1)
	fake_image = tf.reshape(X, shape = (128,128,3))
	plt.imshow(fake_image)
	plt.show()
	"""
if __name__ == '__main__':
	main()


