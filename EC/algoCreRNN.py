import nltk
import nltk.data
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
import random
import scipy.io as scio
import tensorflow as tf
import matplotlib.pyplot as plt

#class for training data set
class Dataset:
	def __init__(self,data,label):    
		self._index_in_epoch = 0
		self._epochs_completed = 0
		self._data = data
		self._label = label
		self._num_examples = data.shape[0]
		pass
	@property
	def data(self):
		return self._data

	@property
	def label(self):
		return self._label

	def next_batch(self,batch_size,shuffle = True):
		start = self._index_in_epoch
		if start == 0 and self._epochs_completed == 0:
			idx = np.arange(0, self._num_examples)
			np.random.shuffle(idx)
			self._data = self.data[idx]
			self._label = self.label[idx]
		if start + batch_size > self._num_examples:
			self._epochs_completed += 1
			rest_num_examples = self._num_examples - start
			data_rest_part = self.data[start:self._num_examples]
			label_rest_part = self.label[start:self._num_examples]
			idx0 = np.arange(0, self._num_examples)
			np.random.shuffle(idx0)
			self._data = self.data[idx0]
			self._label = self.label[idx0]
			start = 0
			self._index_in_epoch = batch_size - rest_num_examples
			end =  self._index_in_epoch  
			data_new_part =  self._data[start:end]  
			label_new_part =  self._label[start:end]  
			return np.concatenate((data_rest_part, data_new_part), axis=0), np.concatenate((label_rest_part, label_new_part), axis=0)
		else:
			self._index_in_epoch += batch_size
			end = self._index_in_epoch
			return self._data[start:end], self._label[start:end]

#load data
example = 'app/mat/example.mat'
data = scio.loadmat(example)
trainIm = data['trainMatrix']
trainLa = data['trainLabour']

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 128

n_inputs = 16
n_steps = 20
n_hidden_units = 128
n_classes = 4

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
	# (28, 128)
	'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
	# (128, 10)
	'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
	# (128, )
	'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
	# (10, )
	'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

def RNN(X, weights, biases):
	X = tf.reshape(X, [-1, n_inputs])

	X_in = tf.matmul(X, weights['in']) + biases['in']
	X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

	if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
		cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
	else:
		cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
	init_state = cell.zero_state(batch_size, dtype=tf.float32)
	
	outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

	if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
		outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
	else:
		outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
	results = tf.matmul(outputs[-1], weights['out']) + biases['out']

	return results

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

trainS=Dataset(trainIm,trainLa)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
saveNet = 'app/my_net/save_RNN.ckpt'
saver.restore(sess, saveNet)

step = 0
while step * batch_size < training_iters:
	batch_xs, batch_ys = trainS.next_batch(128)
	batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
	sess.run([train_op], feed_dict={
		x: batch_xs,
		y: batch_ys,
	})
	if step % 20 == 0:
		print(sess.run(accuracy, feed_dict={
		x: batch_xs,
		y: batch_ys,
		}))
	step += 1
saver.save(sess, saveNet)