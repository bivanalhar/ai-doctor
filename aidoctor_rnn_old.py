import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
method being used : Recurrent Neural Network
In particular, we will use the Long Short Term Memory (LSTM)
"""

#Phase 1 : preprocessing the input, coming from textfile
#1.1 training input file
input_list = []
year_note = []

padding_zero = []
for i in range(73):
	padding_zero.append(0.0) 

f = open('train_arrhytmia.txt', 'r')
for line in f:
	line_1 = line.split(",")
	for i in range(len(line_1)):
		if line_1[i] == '' or line_1[i] == '\n':
			line_1[i] = np.random.normal()
		else:
			# print(line_1[i])
			line_1[i] = np.float32(line_1[i])
	input_list.append(line_1)
f.close()

#here, all the elements of input_list have length 32,
#except that the length of some of them is 33, one of them being the class

pre_data = []
pre_label = []
train_pre_data = []
train_pre_label = []
train_data = []
train_label = []

pos = 0
temp_input = []
temp_label = []
while (pos < len(input_list)):
	id_patient = input_list[pos][0]
	temp_input.append(input_list[pos][2:-1])
	year_note.append(int(input_list[pos][1]) - 2002)
	temp_label.append(float(input_list[pos][-1]))

	# print(temp_label)
	# print(str(id_patient))

	#now processing the labels
	if (pos == len(input_list) - 1) or (id_patient != input_list[pos + 1][0]):
		#which means that the data following this data is belonging to different patient
		pre_data.append(temp_input)
		pre_label.append(temp_label)
		temp_input = []
		temp_label = []

		pos_append = 0
		for i in range(12):
			if i < len(year_note):
				train_pre_data.append(pre_data[0][pos_append])
				if i == len(year_note) - 1:
					if (pre_label[0][pos_append]) == 0:
						train_label.append([1.0, 0.0])
					else:
						train_label.append([0.0, 1.0])
				# train_pre_label.append(pre_label[0][pos_append])
				pos_append += 1
			else:
				train_pre_data.insert(0, padding_zero)
				# train_pre_label.insert(0, [1.0, 0.0])

		train_data.append(train_pre_data)
		# train_label.append(train_pre_label)

		train_pre_data = []
		train_pre_label = []
		year_note = []
		pre_data = []
		pre_label = []

	pos += 1

	# if len(input_list[pos]) == 75:
	# 	temp_input.append(input_list[pos][2:])
	# 	
	# else: 
	# 	temp_input.append(input_list[pos][2:-1])
	# 	year_note.append(int(input_list[pos][1]) - 2002)
	# 	pre_data.append(temp_input)

	# 	if (int(input_list[pos][-1]) == 0):
	# 		train_label.append([1., 0.])
	# 	else:
	# 		train_label.append([0., 1.])
	# 	temp_input = []

	# 	pos_append = 0

	
	# 	train_data.append(train_pre_data)


# print((train_data[4]))
# print(train_label[4])
# util.raiseNotDefined()

#1.2 validating input file
input_list = []

f = open('val_arrhytmia.txt', 'r')
for line in f:
	line_1 = line.split(",")
	for i in range(len(line_1)):
		if line_1[i] == '' or line_1[i] == '\n':
			line_1[i] = np.random.normal()
		else:
			line_1[i] = np.float32(line_1[i])
	input_list.append(line_1)
f.close()

#here, all the elements of input_list have length 32,
#except that the length of some of them is 33, one of them being the class

pre_data = []
pre_label = []
val_pre_data = []
val_pre_label = []
val_data = []
val_label = []

pos = 0
temp_input = []
temp_label = []
while (pos < len(input_list)):
	id_patient = input_list[pos][0]
	temp_input.append(input_list[pos][2:-1])
	year_note.append(int(input_list[pos][1]) - 2002)
	temp_label.append(float(input_list[pos][-1]))

	# print(temp_label)
	# print(str(id_patient))

	#now processing the labels
	if (pos == len(input_list) - 1) or (id_patient != input_list[pos + 1][0]):
		#which means that the data following this data is belonging to different patient
		pre_data.append(temp_input)
		pre_label.append(temp_label)
		temp_input = []
		temp_label = []

		pos_append = 0
		for i in range(12):
			if i < len(year_note):
				val_pre_data.append(pre_data[0][pos_append])
				if i == len(year_note) - 1:
					if (pre_label[0][pos_append]) == 0:
						val_label.append([1.0, 0.0])
					else:
						val_label.append([0.0, 1.0])
				# val_pre_label.append(pre_label[0][pos_append])
				pos_append += 1
			else:
				val_pre_data.insert(0, padding_zero)
				# val_pre_label.insert(0, [1.0, 0.0])

		val_data.append(val_pre_data)
		# val_label.append(val_pre_label)

		val_pre_data = []
		val_pre_label = []
		year_note = []
		pre_data = []
		pre_label = []

	pos += 1

# print(len(val_label))

#1.3 testing input file
input_list = []

f = open('test_arrhytmia.txt', 'r')
for line in f:
	line_1 = line.split(",")
	for i in range(len(line_1)):
		if line_1[i] == '' or line_1[i] == '\n':
			line_1[i] = np.random.normal()
		else:
			line_1[i] = np.float32(line_1[i])
	input_list.append(line_1)
f.close()

#here, all the elements of input_list have length 32,
#except that the length of some of them is 33, one of them being the class

pre_data = []
pre_label = []
test_pre_data = []
test_pre_label = []
test_data = []
test_label = []

pos = 0
temp_input = []
temp_label = []
while (pos < len(input_list)):
	id_patient = input_list[pos][0]
	temp_input.append(input_list[pos][2:-1])
	year_note.append(int(input_list[pos][1]) - 2002)
	temp_label.append(float(input_list[pos][-1]))

	# print(temp_label)
	# print(str(id_patient))

	#now processing the labels
	if (pos == len(input_list) - 1) or (id_patient != input_list[pos + 1][0]):
		#which means that the data following this data is belonging to different patient
		pre_data.append(temp_input)
		pre_label.append(temp_label)
		temp_input = []
		temp_label = []

		pos_append = 0
		for i in range(12):
			if i < len(year_note):
				test_pre_data.append(pre_data[0][pos_append])
				if i == len(year_note) - 1:
					if (pre_label[0][pos_append]) == 0:
						test_label.append([1.0, 0.0])
					else:
						test_label.append([0.0, 1.0])
				# test_pre_label.append(pre_label[0][pos_append])
				pos_append += 1
			else:
				test_pre_data.insert(0, padding_zero)
				# test_pre_label.insert(0, [1.0, 0.0])

		test_data.append(test_pre_data)
		# test_label.append(test_pre_label)

		test_pre_data = []
		test_pre_label = []
		year_note = []
		pre_data = []
		pre_label = []

	pos += 1

# print(len(test_data[0][0]))
# print(len(test_data[4][0]))
#finished with preprocessing the data

# util.raiseNotDefined()

#so now, we have the training data together with the label
#and also we have data and label for validation and test
#however, each data is still string and also there are some
#feature that has no value, we should add some value to it

#phase 2 : building the LSTM model

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('training_epoch', 1000, 'Number of iterations to train')
flags.DEFINE_integer('hidden_nodes', 128, 'Number of nodes in the hidden layer')
flags.DEFINE_integer('batch_size', 128, 'Size of the batch')
flags.DEFINE_float('learning_rate', 0.0001, 'Rate of learning for the optimizer')
flags.DEFINE_float('dropout_rate', 0.2, 'Rate of dropout')
flags.DEFINE_boolean('l2Regularizer', True, 'whether we apply l2Regularize or not')
flags.DEFINE_float('reg_param', 0.1, "parameter to measure the power of regularizer")

def length(sequence):
	used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices = 2))
	length = tf.reduce_sum(used, reduction_indices = 1)
	length = tf.cast(length, tf.int32)
	return length

#defining the variable to hold the input data(30-dimensional data) and label(between yes or no)
data = tf.placeholder(tf.float32, [None, 12, 73])
target = tf.placeholder(tf.float32, [None, 2])

# target_exp = tf.transpose(target, [1, 0, 2])
# target_exp = tf.gather(target_exp, int(target_exp.get_shape()[0]) - 1)

# target = tf.transpose(target, [1,0,2])
# target_exp = tf.gather(target, int(target.get_shape()[0]) - 1)
# target_exp = tf.Print(target_exp, [target_exp], "target_exp = ", summarize = 24)

# target = tf.transpose(target, [1, 0, 2])

#defining the number of nodes in hidden layer (set to 128)
cell = tf.contrib.rnn.core_rnn_cell.LSTMCell(FLAGS.hidden_nodes, forget_bias = 1.0, state_is_tuple=True)
cell = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([cell] * 4, state_is_tuple=True)
cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(cell, output_keep_prob = FLAGS.dropout_rate)

#creating the recurrent neural network with LSTM cells
val, state = tf.nn.dynamic_rnn(cell, data, dtype = tf.float32, sequence_length = length(data))

# val = tf.transpose(val, [1, 0, 2])
# last = tf.gather(val, int(val.get_shape()[0]) - 1)

batch_size_ = tf.shape(val)[0]
max_length = int(val.get_shape()[1])
output_size = int(val.get_shape()[2])
index = tf.range(0, batch_size_) * max_length + (length(data) - 1)
flat = tf.reshape(val, [-1, output_size])
last = tf.gather(flat, index)


#switching the batch size with the sequence size, and then extract the last value
# max_length = int(target.get_shape()[1])
# out_size = int(target.get_shape()[2])
# last = []
# for i in range(int(val.get_shape()[0])):
# 	last.append(tf.gather(val, i))

#defining the weights and biases
weight = tf.Variable(tf.random_normal(shape = [FLAGS.hidden_nodes, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape = [int(target.get_shape()[1])]))

# val = tf.reshape(val, [-1, FLAGS.hidden_nodes])

#defining the prediction, which should be the softmax function
prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
# prediction = tf.reshape(prediction, [-1, max_length, out_size])

# prediction_exp = tf.transpose(prediction, [1, 0, 2])
# prediction_exp = tf.gather(prediction_exp, int(prediction_exp.get_shape()[0]) - 1)
# prediction = tf.Print(prediction, [prediction], 'prediction = ', summarize = 24)

# target = tf.Print(target, [target], 'target = ', summarize = 24)
cross_entropy = target * tf.log(prediction + 1e-10)
cross_entropy = -tf.reduce_sum(cross_entropy)

# mask = tf.sign(tf.reduce_max(tf.abs(target), reduction_indices = 2))
# mask = tf.Print(mask, [tf.argmax(mask, 1)], 'mask = ', summarize = 12)
# mask = tf.sign(tf.reduce_max(tf.abs(target), 2))
# cross_entropy *= mask

# cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices = 1)
# cross_entropy /= tf.cast(length(data), tf.float32)

# cross_entropy /= tf.reduce_sum(mask, reduction_indices = 1)

if FLAGS.l2Regularizer:
	cost = tf.reduce_mean(cross_entropy) + tf.nn.l2_loss(weight) + tf.nn.l2_loss(bias)
	# cost = tf.Print(cost, [cost], 'cost = ', summarize = 24)
else:
	cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate).minimize(cost)

# arg_pred = tf.argmax(prediction, 2)
# arg_pred = tf.Print(arg_pred, [arg_pred], "argmax pred = ", summarize = 60)

correct = tf.equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

#starting the session, by first initializing all the variables
init_op = tf.global_variables_initializer()

display_step = 5

# f = open("170406_result.txt", 'w')
# f.write("Result of the experiment\n\n")

batch_size_list = [128]
hidden_layer_list = [128]
learning_rate_list = [1e-4]
epoch_list_run = [700]
dropout_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.3]
regularizer_parameter = [0.1, 0.01, 0.001]
l2Regularize_list = [True]

count_exp = 1

for batch_size1 in batch_size_list:
	for training_epoch1 in epoch_list_run:
		for learning_rate1 in learning_rate_list:
			for hidden_node1 in hidden_layer_list:
				for dropout_rate1 in dropout_list:
					for l2Reg in l2Regularize_list:
						for reg_param1 in regularizer_parameter:

							FLAGS.batch_size = batch_size1
							FLAGS.hidden_nodes = hidden_node1
							FLAGS.learning_rate = learning_rate1
							FLAGS.training_epoch = training_epoch1
							FLAGS.dropout_rate = dropout_rate1
							FLAGS.l2Regularizer = l2Reg
							FLAGS.reg_param = reg_param1
							epoch_list = []
							cost_list = []	

							print("batch size = " + str(FLAGS.batch_size))
							print("hidden nodes = " + str(FLAGS.hidden_nodes))
							print("learning rate = " + str(FLAGS.learning_rate))
							print("training epoch = " + str(FLAGS.training_epoch))
							print("dropout rate = " + str(1 - FLAGS.dropout_rate))
							print("l2Reg = " + str(FLAGS.l2Regularizer))
							print("reg_param = " + str(FLAGS.reg_param))

							# f.write("setting up the experiment with\n")
							# f.write("batch size = " + str(FLAGS.batch_size) + ", hidden nodes = " + str(FLAGS.hidden_nodes) + ", learning rate = " + str(FLAGS.learning_rate) + "\n")
							# f.write("training epoch = " + str(FLAGS.training_epoch) + ", dropout rate = " + str(1 - FLAGS.dropout_rate) + ", reg_param = " + str(FLAGS.reg_param) + "\n\n")

							with tf.Session() as sess:
								sess.run(init_op)

								for epoch in range(FLAGS.training_epoch):
									# epoch_list.append(epoch + 1)
									ptr = 0
									avg_cost = 0.
									# no_of_batches = int(len(train_data) / FLAGS.batch_size)
									no_of_batches = 1

									for i in range(no_of_batches):
										batch_in, batch_out = train_data[ptr:ptr+FLAGS.batch_size], train_label[ptr:ptr+FLAGS.batch_size]
										ptr += FLAGS.batch_size
										# target_ = sess.run([target], feed_dict = {data : batch_in, target : batch_out})

										_, cost_ = sess.run([optimizer, cost], feed_dict = {data : batch_in, target : batch_out})

										avg_cost += cost_ / no_of_batches
									print("loss function = " + str(avg_cost))
									# cost_list.append(avg_cost)

									# sess.run(target_exp, feed_dict = {data : train_data, target : train_label})
									# sess.run(arg_pred, feed_dict = {data : train_data, target : train_label})

									# if epoch in [9, 19, 49, 99, 199, 299, 499, 699, 999]:
									# 	f.write("During the " + str(epoch+1) + "-th epoch:\n")
									# 	f.write("Training Accuracy = " + str(sess.run(accuracy, feed_dict = {data : train_data, target : train_label})) + "\n")
									# 	f.write("Validation Accuracy = " + str(sess.run(accuracy, feed_dict = {data : val_data, target : val_label})) + "\n")
									# 	f.write("Testing Accuracy = " + str(sess.run(accuracy, feed_dict = {data : test_data, target : test_label})) + "\n\n")
								print("Optimization Finished")

								# plt.plot(epoch_list, cost_list)
								# plt.xlabel("Epoch (dropout = " + str(FLAGS.dropout_rate) + ";l2Reg = " + str(FLAGS.reg_param) + ";epoch = " + str(FLAGS.training_epoch) + ")")
								# plt.ylabel("Cost Function")

								training_accuracy = sess.run(accuracy, feed_dict = {data : train_data, target : train_label})
								validation_accuracy = sess.run(accuracy, feed_dict = {data : val_data, target : val_label})
								testing_accuracy = sess.run(accuracy, feed_dict = {data : test_data, target : test_label})
								
								print("Training Accuracy :", training_accuracy)
								print("Validation Accuracy :", validation_accuracy)
								print("Testing Accuracy :", testing_accuracy)

								# plt.title("Train Acc = " + str(training_accuracy * 100) + "\nTest Acc = " + str(testing_accuracy * 100))

								# plt.savefig("170406 Exp " + str(count_exp) + ".png")

								# plt.clf()

								count_exp += 1

f.close()

########################### will be modified tomorrow ##########################################