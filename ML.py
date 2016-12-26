from parser import CSVParser
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def accuracy_distance(truth, prediction):
	error_sum = 0
	for i in range (0,len(truth)):
		err = abs(truth[i] - prediction[i]) / 20.00
		error_sum += err

	return 1 - (error_sum / len(truth))

def accuracy_distance2(truth, prediction):
	error_sum = 0
	for i in range (0,len(truth)):
		err = abs(truth[i] - prediction[i])
		if err >= 10:
			error_sum += 1.0

	return 1 - (error_sum / len(truth))


csvparser = CSVParser()
data_target = csvparser.parse("student-mat.csv")
grade_x = np.array(data_target[0])
grade_y = np.array(data_target[1])


# Split iris data in train and test data
# A random permutation, to split the data randomly
np.random.seed(1)
indices = np.random.permutation(len(grade_x))

test_size = int(round (len(grade_x) * 0.2)) * -1

grade_x_train = grade_x[indices[:test_size]]
grade_y_train = grade_y[indices[:test_size]]
grade_x_test  = grade_x[indices[test_size:]]
grade_y_test  = grade_y[indices[test_size:]]


# Create and fit a nearest-neighbor classifier
knn = KNeighborsClassifier()
knn.fit(grade_x_train, grade_y_train)
prediction = knn.predict(grade_x_test)
truth =  grade_y_test
print "KNN Accuracy (ours): ", accuracy_distance2(truth, prediction)
print "KNN Accuracy (theirs):", accuracy_score(truth, prediction)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

print truth
print prediction
