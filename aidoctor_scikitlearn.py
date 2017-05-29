from sklearn import svm, metrics, tree, ensemble
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import numpy as np

#pre-processing the data (for the test data and train data)
def process_data(file):
	f = open(file, "r")
	train_data = []
	train_label = []

	for line in f:
		line_split = line.split(",")
		for i in range(len(line_split)):
			if line_split[i] == '' or line_split[i] == '\n':
				line_split[i] = np.abs(np.random.normal())
			else:
				line_split[i] = np.float32(line_split[i])
		train_data.append(line_split[2 : -1])
		train_label.append(line_split[-1])

	return train_data, train_label
	
#pre-processing of the data finished here

########################################################################################

#starting the classification
#Phase 1 : process the data in advance, to get some important features
trainlist = []
testlist = []

trainlist_data, trainlist_label = process_data("training_file/train_arrhytmia.txt")
testlist_data, testlist_label = process_data("testing_file/test_arrhytmia.txt")

#Phase 3 : Train the data and label by using the trainlist
#Phase 3-1 : Train with the Multi-Layer Perceptron
mlplearn = MLPClassifier()
mlplearn.fit(trainlist_data, trainlist_label)

testlist_result = mlplearn.predict(testlist_data)

testlist_proba_ = mlplearn.predict_proba(testlist_data)
testlist_proba = []

count_equal = 0
for i in range(len(testlist_label)):
	testlist_proba.append(testlist_proba_[i][1])
	if testlist_result[i] == testlist_label[i]:
		count_equal += 1.0

print("Training and Testing with Multi-Layer Perceptron")
print("Area Under the Curve is ", metrics.roc_auc_score(testlist_label, testlist_proba))
print("Accuracy for Multi-Layer Perceptron = ", count_equal / len(testlist_label) * 100, "\n")

#Phase 3-2 : Train with the Logistic Regression
loglearn = LogisticRegression()
loglearn.fit(trainlist_data, trainlist_label)

testlist_result = loglearn.predict(testlist_data)

testlist_proba_ = loglearn.predict_proba(testlist_data)
testlist_proba = []

count_equal = 0
for i in range(len(testlist_label)):
	testlist_proba.append(testlist_proba_[i][1])
	if testlist_result[i] == testlist_label[i]:
		count_equal += 1.0

print("Training and Testing with Logistic Regression")
print("Area Under the Curve is ", metrics.roc_auc_score(testlist_label, testlist_proba))
print("Accuracy for Logistic Regression = ", count_equal / len(testlist_label) * 100, "\n")

#Phase 3-3 : Train with the Logistic Regression
logl1learn = LogisticRegression(penalty = 'l1')
logl1learn.fit(trainlist_data, trainlist_label)

testlist_result = logl1learn.predict(testlist_data)

testlist_proba_ = logl1learn.predict_proba(testlist_data)
testlist_proba = []

count_equal = 0
for i in range(len(testlist_label)):
	testlist_proba.append(testlist_proba_[i][1])
	if testlist_result[i] == testlist_label[i]:
		count_equal += 1.0

print("Training and Testing with Logistic Regression (l1 penalty)")
print("Area Under the Curve is ", metrics.roc_auc_score(testlist_label, testlist_proba))
print("Accuracy for Logistic Regression = ", count_equal / len(testlist_label) * 100, "\n")

#Phase 3-4 : Train with the Decision Tree Method
treelearn = tree.DecisionTreeClassifier()
treelearn.fit(trainlist_data, trainlist_label)

testlist_result = treelearn.predict(testlist_data)

testlist_proba_ = treelearn.predict_proba(testlist_data)
testlist_proba = []

count_equal = 0
for i in range(len(testlist_label)):
	testlist_proba.append(testlist_proba_[i][1])
	if testlist_result[i] == testlist_label[i]:
		count_equal += 1.0

print("Training and Testing with Decision Tree Classifier")
print("Area Under the Curve is ", metrics.roc_auc_score(testlist_label, testlist_proba))
print("Accuracy for Decision Tree = ", count_equal / len(testlist_label) * 100, "\n")

#Phase 3-5 : Train with the Random Forest Classifier
forestlearn = ensemble.RandomForestClassifier()
forestlearn.fit(trainlist_data, trainlist_label)

testlist_result = forestlearn.predict(testlist_data)

testlist_proba_ = forestlearn.predict_proba(testlist_data)
testlist_proba = []

count_equal = 0
for i in range(len(testlist_label)):
	testlist_proba.append(testlist_proba_[i][1])
	if testlist_result[i] == testlist_label[i]:
		count_equal += 1.0

print("Training and Testing with Random Forest Classifier")
print("Area Under the Curve is ", metrics.roc_auc_score(testlist_label, testlist_proba))
print("Accuracy for Random Forest = ", count_equal / len(testlist_label) * 100, "\n")

#Phase 3-6 : Train with Bagging Classifier
bagginglearn = ensemble.BaggingClassifier(LogisticRegression())
bagginglearn.fit(trainlist_data, trainlist_label)

testlist_result = bagginglearn.predict(testlist_data)

testlist_proba_ = bagginglearn.predict_proba(testlist_data)
testlist_proba = []

count_equal = 0
for i in range(len(testlist_label)):
	testlist_proba.append(testlist_proba_[i][1])
	if testlist_result[i] == testlist_label[i]:
		count_equal += 1.0

print("Training and Testing with Bagging Classifier")
print("Area Under the Curve is ", metrics.roc_auc_score(testlist_label, testlist_proba))
print("Accuracy for Random Forest = ", count_equal / len(testlist_label) * 100, "\n")

#Phase 3-7 : Train with Voting Classifier
votinglearn = CalibratedClassifierCV(ensemble.VotingClassifier(estimators = 
	[('lr', mlplearn), ('rf', loglearn), ('gnb', treelearn), ('lg1', logl1learn), ('fr', forestlearn), ('bg', bagginglearn)], 
	voting = 'soft'))
votinglearn.fit(trainlist_data, trainlist_label)

testlist_result = votinglearn.predict(testlist_data)

testlist_proba_ = votinglearn.predict_proba(testlist_data)
testlist_proba = []

count_equal = 0
for i in range(len(testlist_label)):
	testlist_proba.append(testlist_proba_[i][1])
	if testlist_result[i] == testlist_label[i]:
		count_equal += 1.0

print("Training and Testing with Voting Classifier")
print("Area Under the Curve is ", metrics.roc_auc_score(testlist_label, testlist_proba))
print("Accuracy for Random Forest = ", count_equal / len(testlist_label) * 100, "\n")