from flask import render_template, flash, redirect, url_for
from app import app
from app.forms import LoginForm, caseForm, chooseForm
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

#create the function of transfering sentence to matrix
def stom(sentence):
    mt = np.zeros(320) + 0.01
    words = word_tokenize(sentence)
    stemmer = PorterStemmer()
    for i in range(len(words)):
        pick = list(stemmer.stem(words[i]))
        for h in range(len(pick)):
            if i*20+h <320:
                mt[i*20+h] += (ord(pick[h])-32)/100
    return mt

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
    # Must have strides[0] = strides[3] = 1
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
W_conv1 = weight_variable([5,5, 1,32]) # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)                                         # output size 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64

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

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
rel_path = 'my_net/save_net.ckpt'
saveNet = os.path.join(script_dir, rel_path)
saver.restore(sess, saveNet)

#test
def test(sentence):
    testM = [stom(sentence)]
    result = ' '
    x2=np.array(testM)
    preResult = sess.run(prediction, feed_dict={xs: x2, keep_prob: 0.5})
    if np.where(preResult==np.max(preResult))[1] == 3:
        result = 'This seems to be a tort of battery.'
    elif np.where(preResult==np.max(preResult))[1] == 2:
        result = 'This seems to be a tort of Assault.'
    elif np.where(preResult==np.max(preResult))[1] == 1:
        result = 'This seems to be a tort of False imprisonment.'
    elif np.where(preResult==np.max(preResult))[1] == 0:
        result = 'This seems to be not a tort.'
    return result

@app.route('/answer',methods=['GET', 'POST'])
def answer():
    type = 'no'
    form = chooseForm()
    if form.validate_on_submit():
        flash('Your choice for defence is: {}'.format(
            form.defence.data))
        flash('Your choice for damages is: {}'.format(
            form.damage.data))
        if (form.defence.data==True):
            flash('Defence is not a tort, I am affraid you have to ask a lawyer for help.')
        elif (form.damage.data==False):
            flash('Got no damage is not a tort, I am affraid you have to ask a lawyer for help.')
        else:
            flash('Now I think you are in a tort')
        type = ''
        return render_template('answer.html', title='Law4U Answer Page', form=form, type=type)
    return render_template('answer.html', title='Law4U Answer Page', form=form, type=type)

@app.route('/',methods=['GET', 'POST'])
@app.route('/index',methods=['GET', 'POST'])
def index():
    form = LoginForm()
    if form.validate_on_submit():
        flash('Login requested for user {}, remember_me={}'.format(
            form.username.data, form.remember_me.data))
        return redirect('/index')
    form2 = caseForm()
    if form2.validate_on_submit():
        flash('Your case is: {}'.format(
            form2.usercase.data))
        result = test(form2.usercase.data)
        flash(result)
        flash('However you could be not in a tort, please answer two simple question:')
        return redirect(url_for('answer'))
    return render_template('index.html', title='Law4U Home Page', form=form,form2=form2)
