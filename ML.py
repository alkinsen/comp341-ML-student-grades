from parser import CSVParser
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def accuracy_distance(truth, prediction):
    error_sum = 0
    for i in range(0, len(truth)):
        err = abs(truth[i] - prediction[i]) / 20.00
        error_sum += err

    return 1 - (error_sum / len(truth))


def accuracy_distance2(truth, prediction):
    error_sum = 0
    for i in range(0, len(truth)):
        err = abs(truth[i] - prediction[i])
        if err >= 10:
            error_sum += 1.0

    return 1 - (error_sum / len(truth))


csvparser = CSVParser()
data_target = csvparser.parse("student-mat.csv")
grade_x_KNN = np.array(data_target[0])
grade_y_KNN = np.array(data_target[1])

#Fing accuracy and coefficients by linear regression
from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit(data_target[0], data_target[1])
coefs = np.array(reg.coef_)


print "The coefficients from linear regression\n", coefs

coef_grade_x = []
for i in range(0, len(grade_x_KNN)):
    coef_features = []
    for j in range(0, len(grade_x_KNN[i])):
        coef_features.append(coefs[j] * grade_x_KNN[i][j])
    coef_grade_x.append(coef_features)

grade_x_LR = np.array(coef_grade_x)

linear_regression_grade_y = []

for i in range(0, len(coef_grade_x)):
    result = sum(coef_grade_x[i])
    result = 0
    if result >=  0.50:
        result = 1






# Split iris data in train and test data
# A random permutation, to split the data randomly
np.random.seed(1)
indices = np.random.permutation(len(grade_x_KNN))

test_size = int(round(len(grade_x_KNN) * 0.2)) * -1

grade_x_train = grade_x_KNN[indices[:test_size]]
grade_y_train = grade_y_KNN[indices[:test_size]]
grade_x_test = grade_x_KNN[indices[test_size:]]
grade_y_test = grade_y_KNN[indices[test_size:]]



# Create and fit a nearest-neighbor classifier
knn = KNeighborsClassifier()
knn.fit(grade_x_train, grade_y_train)
prediction = knn.predict(grade_x_test)
truth = grade_y_test
print "KNN Accuracy (ours): ", accuracy_distance2(truth, prediction)
print "KNN Accuracy (theirs):", accuracy_score(truth, prediction)


print prediction
print truth




#Plotting the coefficients from linear regression
features = ["school", "sex", "age", "address", "famsize", "pstatus", "medu", "fedu", "traveltime", "studytime",
            "failures",
            "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic", "famrel",
            "freetime", "goout", "dalc", "walc",
            "health", "absences"]

import matplotlib.pyplot as plt

# plt.axis([0,25,-0.1,0.1])
plt.xlabel('Features')
plt.ylabel('Weight')
abscoefs = [abs(x) for x in coefs]
plt.xticks(np.array(range(0, len(features))), features, rotation=90)
plt.plot(np.array(range(0, len(features))), abscoefs, 'r*')
plt.show()
