import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

"""
Method being used : Convolutional Neural Network
Here we are adapting the method being used in the paper titled
Convolutional Neural Network for Sentence Classification
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

			for i in range(6):
				temp_feat.insert(0, padding_zero)

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

#Phase 2 : Building up the CNN Model

#defining the hyperparameter first
training_epoch = 1000
hidden_nodes = 128
batch_size = 128
learning_rate = 0.0001
dropout_rate = 0.2
l2_regularize = True
reg_param = 0.1
kernel_size = 4

#defining up the variable used for holding the input data and the label
data = tf.placeholder(tf.float32, [None, 18, 73])
target_12 = tf.placeholder(tf.float32, [None, 12, 2])
target_1 = tf.placeholder(tf.float32, [None, 2])

weight_1 = tf.Variable(tf.random_normal(shape = [kernel_size, int(data.get_shape()[2]), 32]))
bias_1 = tf.Variable(tf.constant(0.1, shape = [32]))

weight_2 = tf.Variable(tf.random_normal(shape = [kernel_size, 32, 64]))
bias_2 = tf.Variable(tf.constant(0.1, shape = [64]))

weight_final = tf.Variable(tf.random_normal(shape = [1, 64, 2]))
bias_final = tf.Variable(tf.constant(0.1, shape = [2]))

conv1 = tf.nn.conv1d(data, weight_1, stride = 1, padding = 'VALID')
# print(conv1.get_shape().as_list())
conv1 = tf.nn.relu(conv1 + bias_1)
# print(conv1.get_shape())

conv2 = tf.nn.conv1d(conv1, weight_2, stride = 1, padding = 'VALID')
conv2 = tf.nn.relu(conv2 + bias_2)
# print(conv2.get_shape())

conv_final = tf.nn.conv1d(conv2, weight_final, stride = 1, padding = 'VALID')
conv_final = tf.nn.softmax(tf.nn.relu(conv_final + bias_final))

conv_last = tf.transpose(conv_final, [1, 0, 2])
conv_last = tf.gather(conv_last, int(conv_last.get_shape()[0]) - 1)

# print(conv_final.get_shape())
# conv_final = tf.Print(conv_final, [conv_final], summarize = 24)

#getting the last element of the conv_final

# print(conv_last.get_shape())
# conv_last = tf.Print(conv_last, [conv_last], summarize = 32)

cross_entropy = -tf.reduce_sum(target_12 * tf.log(tf.clip_by_value(conv_final, 1e-10, 1.0)), [1, 2])

if l2_regularize:
	loss = tf.reduce_mean(cross_entropy) + reg_param*(tf.nn.l2_loss(weight_1)+tf.nn.l2_loss(bias_1)+tf.nn.l2_loss(weight_2)+tf.nn.l2_loss(bias_2)+tf.nn.l2_loss(weight_final)+tf.nn.l2_loss(bias_final))
else:
	loss = tf.reduce_mean(cross_entropy)

# loss = tf.losses.softmax_cross_entropy(onehot_labels = target, logits = logits) + reg_param*(tf.nn.l2_loss(weight_1)+tf.nn.l2_loss(bias_1)+tf.nn.l2_loss(weight_2)+tf.nn.l2_loss(bias_2))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

correct = tf.equal(tf.argmax(target_1, 1), tf.argmax(conv_last, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

#initializing all the trainable parameters here
init_op = tf.global_variables_initializer()

# f = open("170423_result_cnnseq.txt", 'w')
# f.write("Result of the experiment\n\n")

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

							# f.write("setting up the experiment with\n")
							# f.write("batch size = " + str(batch_size) + ", hidden nodes = " + str(hidden_nodes) + ", learning rate = " + str(learning_rate) + "\n")
							# f.write("training epoch = " + str(training_epoch) + ", dropout rate = " + str(1 - dropout_rate) + ", reg_param = " + str(reg_param) + "\n\n")

							with tf.Session() as sess:
								sess.run(init_op)
								# sess.run(conv_final, feed_dict = {data : train_data, target : train_label_test})

								for epoch in range(training_epoch):
									# epoch_list.append(epoch + 1)
									ptr = 0
									avg_cost = 0.
									# no_of_batches = int(len(train_data) / batch_size)
									no_of_batches = 1

									for i in range(no_of_batches):
										batch_in, batch_out, batch_test = train_data[ptr:ptr+batch_size], train_label_train[ptr:ptr+batch_size], train_label_test[ptr:ptr+batch_size]
										ptr += batch_size
										# target_ = sess.run([target], feed_dict = {data : batch_in, target : batch_out})

										_, cost_ = sess.run([optimizer, loss], feed_dict = {data : batch_in, target_12 : batch_out})

										avg_cost += cost_ / no_of_batches
									print("loss function = " + str(avg_cost))
									# cost_list.append(avg_cost)

									# sess.run(target_exp, feed_dict = {data : train_data, target : train_label})
									# sess.run(arg_pred, feed_dict = {data : train_data, target : train_label})

									# if epoch in [9, 19, 49, 99, 199, 299, 499, 699, 999]:
									# 	f.write("During the " + str(epoch+1) + "-th epoch:\n")
									# 	f.write("Training Accuracy = " + str(sess.run(accuracy, feed_dict = {data : train_data, target_1 : train_label_test})) + "\n")
									# 	f.write("Validation Accuracy = " + str(sess.run(accuracy, feed_dict = {data : val_data, target_1 : val_label})) + "\n")
									# 	f.write("Testing Accuracy = " + str(sess.run(accuracy, feed_dict = {data : test_data, target_1 : test_label})) + "\n\n")
								print("Optimization Finished")

								# plt.plot(epoch_list, cost_list)
								# plt.xlabel("Epoch (dropout = " + str(dropout_rate) + "learning rate = " + str(learning_rate) + ";epoch = " + str(training_epoch) + ")")
								# plt.ylabel("Cost Function")

								training_accuracy = sess.run(accuracy, feed_dict = {data : train_data[0:128-1], target_1 : train_label_test[0:128-1]})
								validation_accuracy = sess.run(accuracy, feed_dict = {data : val_data[0:128-1], target_1 : val_label[0:128-1]})
								testing_accuracy = sess.run(accuracy, feed_dict = {data : test_data[0:128-1], target_1 : test_label[0:128-1]})
								
								print("Training Accuracy :", training_accuracy)
								print("Validation Accuracy :", validation_accuracy)
								print("Testing Accuracy :", testing_accuracy)

								# plt.title("Train Acc = " + str(training_accuracy * 100) + "\nTest Acc = " + str(testing_accuracy * 100))

								# plt.savefig("170423_cnnseq Exp " + str(count_exp) + ".png")

								# plt.clf()

								count_exp += 1
# f.close()