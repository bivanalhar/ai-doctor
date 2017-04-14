import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
Method being used : Recurrent Neural Network
Unit Cell Type : Long Short Term Memory (LSTM)
"""

#Phase 1 : Preprocessing the input
"""
Summary of the file : Each line contains a patient's information,
the information about the year as well as 73 features, together with
the label (whether that particular patient got the disease or not)
"""

padding_zero = [0.0] * 73 #for the padding zero

#function special for the one-hot encoding
def one_hot(label):
	if label == 0:
		return [1.0, 0.0]
	else:
		return [0.0, 1.0]

#this function is about to process the input file
def process_input(file, train_stat = True):
	f = open(file, 'r') #opening the file
	input_list = []
	train_data = []
	train_label = []
	temp_traindata = []

	for line in f: #processing every line of the content of the input file
		line_split = line.split(",")
		#splitting lines into array of useful feature
		for i in range(len(line_split)):
			if line_split[i] == '' or line_split[i] == '\n':
				line_split[i] = np.abs(np.random.normal()) #note that value of the features cannot be negative
			else:
				line_split[i] = np.float32(line_split[i])		
		input_list.append(line_split)
	f.close() #closing the file, meaning that we have finished extracting the info from the file
	
	"""
	Info for the input_list : it has multiple info of length 76, with the first element being the ID of the
	patient, the second one being the year, then followed by 73 features and being ended by the label (0/1)
	"""

	#begin processing the input list
	cur_posit = 0 #denoting the current position the element is being pointed at

	#initialization of the temporary bucket for feature and label
	temp_feat = []
	temp_label = []

	while (cur_posit < len(input_list)):

		#collecting the full information about the patient, storing it to the appropriate place
		assert (len(input_list[cur_posit]) == 76) #just to make sure the length of line is okay
		feat_patient = input_list[cur_posit][2:-1] #info about all the features for that line
		id_patient = input_list[cur_posit][0] #info about the patient's identity number
		label_patient = input_list[cur_posit][-1]

		#begin putting on the list to be trained/tested
		if (cur_posit != len(input_list) - 1) and (id_patient == input_list[cur_posit + 1][0]):
			#if the data following the currently explored one still belongs to the same patient
			#in this case, simply collect those feature and labelling to the temporary bucket of
			#feature and label
			temp_feat.append(feat_patient)
			temp_label.append(one_hot(label_patient))

		else:
			#reaching the end of the data for a particular patient. In this case, putting that last
			#info to the temporary bucket first
			temp_feat.append(feat_patient)
			temp_label.append(one_hot(label_patient))

			#after that push into the training data, but with padding zero first
			for i in range(12 - len(temp_feat)):
				temp_feat.insert(0, padding_zero)
				temp_label.insert(0, [1.0, 0.0])

			train_data.append(temp_feat)

			if train_stat:
				train_label.append(temp_label)
			else:
				train_label.append(temp_label[-1])

			#lastly, flush all the temporary bucket so that it will be used as new again
			temp_feat = []
			temp_label = []
		cur_posit += 1
	return train_data, train_label

#collecting all the data for the training, validation and testing
train_data, train_label_train = process_input('training_file/train_arrhytmia.txt', train_stat = True) #for the training sake
_, train_label_test = process_input('training_file/train_arrhytmia.txt', train_stat = False) #for the testing sake
val_data, val_label = process_input('validation_file/val_arrhytmia.txt', train_stat = False)
test_data, test_label = process_input('testing_file/test_arrhytmia.txt', train_stat = False)

# print(train_label.count([0.0, 1.0]))
# print(len(train_label))
# print(len(train_label[0]))

#################################PHASE 1 FINISHED#####################################

#Phase 2 : Building up the LSTM Model

#defining the hyperparameter first
training_epoch = 1000
hidden_nodes = 128
batch_size = 128
learning_rate = 0.0001
dropout_rate = 0.2
l2_regularize = True
reg_param = 0.1

#defining up the variable used for holding the input data and the label
data = tf.placeholder(tf.float32, [None, 12, 73])
target_12 = tf.placeholder(tf.float32, [None, 12, 2])
target_1 = tf.placeholder(tf.float32, [None, 2])

cell = tf.contrib.rnn.core_rnn_cell.LSTMCell(hidden_nodes, forget_bias = 1.0, state_is_tuple=True)
cell = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([cell] * 4, state_is_tuple=True)
cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(cell, output_keep_prob = dropout_rate)

val, state = tf.nn.dynamic_rnn(cell, data, dtype = tf.float32)

max_length = int(target_12.get_shape()[1])
num_classes = int(target_12.get_shape()[2])

val_cost = tf.reshape(val, [-1, hidden_nodes])

#defining the initialized value for the weight and bias
weight = tf.Variable(tf.random_normal(shape = [hidden_nodes, num_classes]))
bias = tf.Variable(tf.constant(0.1, shape = [num_classes]))

#this is for the testing sake
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

prediction_test = tf.nn.softmax(tf.matmul(last, weight) + bias)

#now defining the prediction vector, which should be the softmax function after being multiplied
#by W and b, then we define the cross entropy
prediction_cost = tf.nn.softmax(tf.matmul(val_cost, weight) + bias)
prediction_cost = tf.reshape(prediction_cost, [-1, max_length, num_classes])
cross_entropy = -tf.reduce_sum(target_12 * tf.log(tf.clip_by_value(prediction_cost,1e-10,1.0)), [1, 2])

if l2_regularize:
	cost = tf.reduce_mean(cross_entropy) + tf.nn.l2_loss(weight) + tf.nn.l2_loss(bias)
else:
	cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

#measurement of the accuracy value of the dataset
correct = tf.equal(tf.argmax(target_1, 1), tf.argmax(prediction_test, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

#initializing all the trainable parameters here
init_op = tf.global_variables_initializer()

f = open("170414_result.txt", 'w')
f.write("Result of the experiment\n\n")

batch_size_list = [128]
hidden_layer_list = [128]
learning_rate_list = [1e-4]
epoch_list_run = [1000]
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

							batch_size = batch_size1
							hidden_nodes = hidden_node1
							learning_rate = learning_rate1
							training_epoch = training_epoch1
							dropout_rate = dropout_rate1
							l2_regularize = l2Reg
							reg_param = reg_param1
							epoch_list = []
							cost_list = []	

							print("batch size = " + str(batch_size))
							print("hidden nodes = " + str(hidden_nodes))
							print("learning rate = " + str(learning_rate))
							print("training epoch = " + str(training_epoch))
							print("dropout rate = " + str(1 - dropout_rate))
							print("l2Reg = " + str(l2_regularize))
							print("reg_param = " + str(reg_param))

							f.write("setting up the experiment with\n")
							f.write("batch size = " + str(batch_size) + ", hidden nodes = " + str(hidden_nodes) + ", learning rate = " + str(learning_rate) + "\n")
							f.write("training epoch = " + str(training_epoch) + ", dropout rate = " + str(1 - dropout_rate) + ", reg_param = " + str(reg_param) + "\n\n")

							with tf.Session() as sess:
								sess.run(init_op)

								for epoch in range(training_epoch):
									epoch_list.append(epoch + 1)
									ptr = 0
									avg_cost = 0.
									no_of_batches = int(len(train_data) / batch_size)
									# no_of_batches = 3

									for i in range(no_of_batches):
										batch_in, batch_out, batch_test = train_data[ptr:ptr+batch_size], train_label_train[ptr:ptr+batch_size], train_label_test[ptr:ptr+batch_size]
										ptr += batch_size
										# target_ = sess.run([target], feed_dict = {data : batch_in, target : batch_out})

										_, cost_ = sess.run([optimizer, cost], feed_dict = {data : batch_in, target_12 : batch_out, target_1 : batch_test})

										avg_cost += cost_ / no_of_batches
									# print("loss function = " + str(avg_cost))
									cost_list.append(avg_cost)

									# sess.run(target_exp, feed_dict = {data : train_data, target : train_label})
									# sess.run(arg_pred, feed_dict = {data : train_data, target : train_label})

									if epoch in [9, 19, 49, 99, 199, 299, 499, 699, 999]:
										f.write("During the " + str(epoch+1) + "-th epoch:\n")
										f.write("Training Accuracy = " + str(sess.run(accuracy, feed_dict = {data : train_data, target_1 : train_label_test})) + "\n")
										f.write("Validation Accuracy = " + str(sess.run(accuracy, feed_dict = {data : val_data, target_1 : val_label})) + "\n")
										f.write("Testing Accuracy = " + str(sess.run(accuracy, feed_dict = {data : test_data, target_1 : test_label})) + "\n\n")
								print("Optimization Finished")

								plt.plot(epoch_list, cost_list)
								plt.xlabel("Epoch (dropout = " + str(dropout_rate) + ";l2Reg = " + str(reg_param) + ";epoch = " + str(training_epoch) + ")")
								plt.ylabel("Cost Function")

								training_accuracy = sess.run(accuracy, feed_dict = {data : train_data, target_1 : train_label_test})
								validation_accuracy = sess.run(accuracy, feed_dict = {data : val_data, target_1 : val_label})
								testing_accuracy = sess.run(accuracy, feed_dict = {data : test_data, target_1 : test_label})
								
								print("Training Accuracy :", training_accuracy)
								print("Validation Accuracy :", validation_accuracy)
								print("Testing Accuracy :", testing_accuracy)

								plt.title("Train Acc = " + str(training_accuracy * 100) + "\nTest Acc = " + str(testing_accuracy * 100))

								plt.savefig("170414 Exp " + str(count_exp) + ".png")

								plt.clf()

								count_exp += 1