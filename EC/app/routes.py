from flask import render_template, flash, redirect, url_for
from app import app
from app.forms import LoginForm, caseForm, chooseForm, chooseTort, identify
import nltk
import nltk.data
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
import random
import scipy.io as scio
import tensorflow as tf
import matplotlib.pyplot as plt

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

#create the function of transfering labour into code
def stoc(labour):
    mt = np.zeros(4)
    code = bin(int(labour)).lstrip('0b').zfill(4)
    for i in range(4):
        mt[i] = int(code[i])
    return mt

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
	'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
	'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
	'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
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
		outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))
	else:
		outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
	results = tf.matmul(outputs[-1], weights['out']) + biases['out']

	return results

pred = RNN(x, weights, biases)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
saveNet = '/root/EC/app/my_net/save_RNN.ckpt'
saver.restore(sess, saveNet)

def test(sentence):
	testM = [0]
	for i in range(0,batch_size) :
		testM.append(stom(sentence))
	del testM[0]
	result = ' '
	batch_xs=np.array(testM)
	batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
	preResult = sess.run([pred], feed_dict={x:batch_xs})
	print(sentence)
	if np.where(preResult==np.max(preResult))[2][0] == 3:
		result = 'This seems to be a tort of battery. Which means this has direct contact, physical interference and is accompanied by fault.'
	elif np.where(preResult==np.max(preResult))[2][0] == 2:
		result = 'This seems to be a tort of Assault. Which means this is direct act by defendant, in mind of the plaintiff and must be intension to cause apprehension'
	elif np.where(preResult==np.max(preResult))[2][0] == 1:
		result = 'This seems to be a tort of False imprisonment. Which means this is direct voluntary act, total deprivation od free movement and without lawful justifaction.'
	elif np.where(preResult==np.max(preResult))[2][0] == 0:
		result = 'This seems to be not a tort.'
	return result

example = '/root/EC/app/mat/example.mat'
data = scio.loadmat(example)
mainMT = np.array(data['trainMatrix']).tolist()
mainLT = np.array(data['trainLabour']).tolist()
count=0

@app.route('/answer?result=<result>',methods=['GET', 'POST'])
def answer(result):
	type = 'no'
	type2 = ''
	form = chooseForm()
	if form.validate_on_submit():
		flash('Your choice for defence is: {}'.format(
			form.defence.data))
		flash('Your choice for damages is: {}'.format(
			form.damage.data))
		if (form.defence.data==True):
			flash('Self defence is not a tort and it is hard to identify. I am affraid you have to ask a lawyer for help.')
		elif (form.damage.data==False):
			flash('Got no damage is not a tort and even not a case. I am affraid you have to ask a lawyer for help.')
		else:
			flash('Now I think you are in a tort')
		type = ''
		return render_template('answer.html', title='Law4U Answer Page', form=form, form2=form2, type=type)
	form2 = chooseTort()
	if form2.validate_on_submit():
		b = form2.battery.data
		a = form2.assult.data
		f = form2.fault.data
		n = form2.notT.data
		if b==True:
			mainMT.append(stom(result))
			mainLT.append(stoc(1))
		if a==True:
			mainMT.append(stom(result))
			mainLT.append(stoc(2))
		if f==True:
			mainMT.append(stom(result))
			mainLT.append(stoc(4))
		if n==True:
			mainMT.append(stom(result))
			mainLT.append(stoc(8))
		count=count+1
		if count == 128:
			Sdata = {}
			storeTrainMatrix = np.array(mainMT)
			Sdata['trainMatrix'] = storeTrainMatrix
			storeTrainLabour = np.array(mainLT)
			Sdata['trainLabour'] = storeTrainLabour
			scio.savemat(example, Sdata)
			count = 0
		return render_template('answer.html', title='Law4U Answer Page', form=form,form2=form2, type=type)
	form3 = identify()
	if form3.validate_on_submit():
		c1 = form3.c1.data
		c2 = form3.c2.data
		c3 = form3.c3.data
		if (c1==True) and (c2==True) and (c3==True):
			mainMT.append(stom(result))
			mainLT.append(stoc(1))
		count=count+1
		if count == 128:
			Sdata = {}
			storeTrainMatrix = np.array(mainMT)
			Sdata['trainMatrix'] = storeTrainMatrix
			storeTrainLabour = np.array(mainLT)
			Sdata['trainLabour'] = storeTrainLabour
			scio.savemat(example, Sdata)
			count = 0
		return render_template('answer.html', title='Law4U Answer Page', form=form,form2=form2, type=type)
	return render_template('answer.html', title='Law4U Answer Page', form=form,form2=form2, type=type)

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
		case = form2.usercase.data
		flash('Your case is: {}'.format(
			case))
		result = test(case)
		flash(result)
		flash('This result is come from the similarity of other cases. So you might think your input did not contain these things.')
		flash('You could also be not in a tort, please answer two simple question:')
		return redirect(url_for('answer', result=result))
	return render_template('index.html', title='Law4U Home Page', form=form,form2=form2)

@app.route('/contact')
def contact():
	return render_template('contact.html', title='Law4U connect Page')

@app.route('/lawyer')
def lawyer():
	return render_template('lawyer.html', title='Law4U contact Page')