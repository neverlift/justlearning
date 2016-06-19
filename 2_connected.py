from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	train_dataset = save['train_dataset']
	train_labels = save['train_labels']
	valid_dataset = save['valid_dataset']
	valid_labels = save['valid_labels']
	test_dataset = save['test_dataset']
	test_labels = save['test_labels']
	del save  # hint to help gc free up memory
	print('Training set', train_dataset.shape, train_labels.shape)
	print('Validation set', valid_dataset.shape, valid_labels.shape)
	print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10
def reformat(dataset, labels):
	dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
	# Map 0 to [1, 0, 0...] 1 to [0, 1, 0, ...]
	transposed_labels = np.array([labels]).T
	labels = (np.arange(num_labels) == transposed_labels).astype(np.float32)
	return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# need a subset to go faster
train_subset = 10000

graph = tf.Graph()
with graph.as_default():

	# Input the data
	# Load training, validation, and test data into constants
	tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
	tf_train_labels = tf.constant(train_labels[:train_subset, :])
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)

	# Variables, aka the parameters to be trained.
	# Weights are initialized according to truncated normal
	# Biases initialized to 0
	weights = tf.Variable(
		tf.truncated_normal([image_size * image_size, num_labels]))
	biases = tf.Variable(tf.zeros([num_labels]))

	# Multiply weights by input, and add biases.
	# Then average cross entropy to find the loss
	logits = tf.matmul(tf_train_dataset, weights) + biases
	loss = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

	# Optimize by finding the minimum of loss using gradient descent
	optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

	# Report accuracy while training
	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
	test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
	