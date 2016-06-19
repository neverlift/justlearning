import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf
from time import time

# Settings
LEARNING_RATE = 1e-4

# Should return 0.99 accuracy
TRAINING_ITERATIONS = 20000

DROPOUT = 0.5
BATCH_SIZE = 50

# Set to 0 to train on all available data
VALIDATION_SIZE = 2000



data = pd.read_csv('train.csv')

# print('data({0[0]}, {0[1]})'.format(data.shape))
# equivalent to print('data(' + str(data.shape[0]) + ',' + str(data.shape[1]) + ")")
# print(data.head())

# Image number to output
# IMAGE_TO_DISPLAY = np.random.randint(0,data.shape[0] - 1)

images = data.iloc[:,1:].values # strip out the labels
#print(images)

images = images.astype(np.float)

# Convert from [0:255] => [0.0:1.0]
images = np.multiply(images, 1.0/255.0)

#print('images({0[0]}, {0[1]})'.format(images.shape))

image_size = images.shape[1]
#print('image_size => {0}'.format(image_size))

# This step only works if every image is square
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

#print('image_width = ' + str(image_width) + ' image_height = ' + str(image_height))

def display(img):
	# (784) => (28,28)
	one_image = img.reshape(image_width, image_height)
	
	plt.axis('off')
	plt.imshow(one_image, cmap=cm.binary)
	plt.show()
# show a random image
# print(IMAGE_TO_DISPLAY)
# display(images[IMAGE_TO_DISPLAY])

labels_flat = data[[0]].values.ravel()

# print('labels_flat({0})'.format(len(labels_flat)))
# print('labels_flat([{0}]) => {1}'.format(IMAGE_TO_DISPLAY, labels_flat[IMAGE_TO_DISPLAY]))
labels_count = np.unique(labels_flat).shape[0]
#print('labels_count => {0}'.format(labels_count))

# convert to one hot vectors
# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
# ...
# 9 => [0 0 0 0 0 0 0 0 0 1]

def dense_to_one_hot(labels_dense, num_classes):
	one_hots = []
	for label in labels_dense:
		one_hot = np.zeros(num_classes)
		one_hot[label] = 1
		one_hots.append(one_hot)
	return one_hots
	
def dense_2_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

labels = dense_2_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)

# print(labels[IMAGE_TO_DISPLAY])

validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')