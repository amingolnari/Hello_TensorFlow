"""
@auther: Amin Golnari
Shahrood University of Technology - IRAN
Code Name: Hello TensorFlow
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import time

Data = input_data.read_data_sets("MNIST/", one_hot = True)
print("# Train data dimension: (%d, %d)" %
      (Data.train.images.shape[0], Data.train.images.shape[1]))
print("# Validation data dimension: (%d, %d)" %
      (Data.validation.images.shape[0], Data.validation.images.shape[1]))
print("# Test data dimension: (%d, %d)" %
      (Data.test.images.shape[0], Data.test.images.shape[1]))

InputShape = 784
NumClass = 10
lr = 0.45
Epochs = 10
Batch = 100
NumBatch = (int)(Data.train.num_examples/Batch)
train, val = [], []

Input = tf.placeholder(tf.float32, [None, InputShape])
Weight = tf.Variable(tf.random_uniform([InputShape, NumClass]))
Bias = tf.Variable(tf.zeros([NumClass]))
Output = tf.nn.softmax(tf.matmul(Input, Weight) + Bias)
Label = tf.placeholder(tf.float32, [None, NumClass])
CrossEntropy = tf.reduce_mean(-tf.reduce_sum(Label * tf.log(Output), reduction_indices = [1]))
Optimizer = tf.train.GradientDescentOptimizer(lr).minimize(CrossEntropy)

Predict = tf.equal(tf.argmax(Output, 1), tf.argmax(Label, 1))
Accuracy = tf.reduce_mean(tf.cast(Predict, tf.float32))

Init = tf.global_variables_initializer()

with tf.Session() as Sess:
	Sess.run(Init)
	for epoch in range(Epochs):
		AvgCost = 0.0
		StartTime = time.time()
		for i in range(NumBatch):
			BatchI, BatchT = Data.train.next_batch(Batch)
			_, cost = Sess.run([Optimizer, CrossEntropy], feed_dict = {Input: BatchI, Label: BatchT})
			AvgCost += cost / Batch
		EndTime = time.time()
		train.append(Accuracy.eval({Input: Data.train.images, Label: Data.train.labels}) * 100)
		val.append(Accuracy.eval({Input: Data.validation.images, Label: Data.validation.labels}) * 100)
		print('# Epoch : %d/%d | Test Accuracy : %.2f | Validation Accuracy : %.2f | Cost : %.6f | Time : %.4fs' %
		      (epoch+1, Epochs,
		       train[epoch],
		       val[epoch],
		       AvgCost, (EndTime - StartTime)))
	plt.figure(1)
	plt.plot(train, label = 'Train')
	plt.plot(val, label = 'Validation')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.title('MNIST')
	plt.grid()
	plt.show()
	Evaluate = Accuracy.eval({Input: Data.test.images, Label: Data.test.labels})
	print("# Test Accuracy: %.2f" % (Evaluate*100))
