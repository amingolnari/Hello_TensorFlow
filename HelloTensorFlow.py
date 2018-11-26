
"""
@auther: Amin Golnari
Shahrood University of Technology
Code Name: Hello TensorFlow
"""

# Inport TensorFlow and MNIST Data
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import time

# Load (or Download for First) Data From Path
Data = input_data.read_data_sets("/temp/MNIST/", one_hot = True)
print("# Train data dimension: (%d, %d)" %
      (Data.train.images.shape[0], Data.train.images.shape[1]))
print("# Validation data dimension: (%d, %d)" %
      (Data.validation.images.shape[0], Data.validation.images.shape[1]))
print("# Test data dimension: (%d, %d)" %
      (Data.test.images.shape[0], Data.test.images.shape[1]))

# Initialize Parameters
InputShape = 784 # MNIST Input Shape (Image Shape: 28*28 = 784)
NumClass = 10 # MNIST Total Classes (0-9 digits)
lr = 0.45 # Learning Rate
Epochs = 10 # Training Epochs
Batch = 100 # Batch Train
NumBatch = (int)(Data.train.num_examples/Batch) # Total Batches
train, val = [], []

# Create Tensors
Input = tf.placeholder(tf.float32, [None, InputShape]) # TF Graph Input
Weight = tf.Variable(tf.random_uniform([InputShape, NumClass]))
Bias = tf.Variable(tf.zeros([NumClass]))
Output = tf.nn.softmax(tf.matmul(Input, Weight) + Bias) # TF Graph Output
Label = tf.placeholder(tf.float32, [None, NumClass])

# Define Cost and Optimizer
CrossEntropy = tf.reduce_mean(-tf.reduce_sum(Label * tf.log(Output), reduction_indices = [1]))
Optimizer = tf.train.GradientDescentOptimizer(lr).minimize(CrossEntropy)

Predict = tf.equal(tf.argmax(Output, 1), tf.argmax(Label, 1))
Accuracy = tf.reduce_mean(tf.cast(Predict, tf.float32))

# Initializing All the Variables
Init = tf.global_variables_initializer()


with tf.Session() as Sess:
	Sess.run(Init)
	# Train Epochs
	for epoch in range(Epochs):
		AvgCost = 0.0
		StartTime = time.time()
		# Training with All Batches
		for i in range(NumBatch):
			BatchI, BatchT = Data.train.next_batch(Batch)
			# Run Optimizer and Cost to Get Loss Value on Batch
			_, cost = Sess.run([Optimizer, CrossEntropy], feed_dict = {Input: BatchI, Label: BatchT})
			# Compute avg Loss
			AvgCost += cost / Batch
		EndTime = time.time()
		# Get Train and Validation Accuracy
		train.append(Accuracy.eval({Input: Data.train.images, Label: Data.train.labels}) * 100)
		val.append(Accuracy.eval({Input: Data.validation.images, Label: Data.validation.labels}) * 100)
		# Display Informations
		print('# Epoch : %d/%d | Train Accuracy : %.2f | Validation Accuracy : %.2f | Cost : %.6f | Time : %.4fs' %
		      (epoch+1,
		       Epochs,
		       train[epoch],
		       val[epoch],
		       AvgCost, (EndTime - StartTime)))
	# Plot Accuracy
	plt.figure(1)
	plt.plot(train, label = 'Train')
	plt.plot(val, label = 'Validation')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.title('MNIST')
	plt.grid()
	plt.legend()
	plt.show()
	
	# Evaluate Test Accuracy
	Evaluate = Accuracy.eval({Input: Data.test.images, Label: Data.test.labels})
	print("# Test Accuracy: %.2f" % (Evaluate*100))
