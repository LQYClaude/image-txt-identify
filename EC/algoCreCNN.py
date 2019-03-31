import nltk
import nltk.data
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
import random
import scipy.io as scio
import tensorflow as tf
import matplotlib.pyplot as plt
import os

#set script_dir
script_dir = os.path.dirname(__file__)

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
rel_path = 'app/mat/example.mat'
example = os.path.join(script_dir, rel_path)
data = scio.loadmat(example)
trainIm = data['trainMatrix']
trainLa = data['trainLabour']

#function for compare accuracy
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

#function for w
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#function for bias
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#function for conv2d
def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#function for max pool
def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#train
xs = tf.placeholder(tf.float32, [None, 320])
ys = tf.placeholder(tf.float32, [None, 4])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 20, 16, 1])

## conv1 layer ##
W_conv1 = weight_variable([5,5, 1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

## conv2 layer ##
W_conv2 = weight_variable([5,5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

## fc1 layer ##
W_fc1 = weight_variable([5*4*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 5*4*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, 4])
b_fc2 = bias_variable([4])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()

saver = tf.train.Saver()
rel_path = 'app/my_net/save_net.ckpt'
saveNet = os.path.join(script_dir, rel_path)
saver.restore(sess, saveNet)

trainS=Dataset(trainIm,trainLa)

lastCom = 0
i=0
while i < 202:
    x,y = trainS.next_batch(100)
    sess.run(train_step, feed_dict={xs: x, ys: y, keep_prob: 0.5})
    if (i == 200) and ((compute_accuracy(x,y)<0.8) or (lastCom<0.8)):
        print('re')
        i=0
        saver.save(sess, saveNet)
    if i % 100 == 0:
        print(compute_accuracy(x,y))
        lastCom = compute_accuracy(x,y)
    i=i+1
