from parser import CSVParser
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import linear_model
import matplotlib.pyplot as plt


def accuracy_distance(truth, prediction):
    error_sum = 0
    for i in range(0, len(truth)):
        err = abs(truth[i] - prediction[i])
        error_sum += err

    return 1 - (error_sum / len(truth))


def accuracy_distance2(truth, prediction):
    error_sum = 0
    for i in range(0, len(truth)):
        err = abs(truth[i] - prediction[i])
        if err >= 10:
            error_sum += 1.0

    return 1 - (error_sum / len(truth))


def KNNprediction(grade_x, grade_y):
    np.random.seed(1)
    indices = np.random.permutation(len(grade_x))
    test_size = int(round(len(grade_x) * 0.2)) * -1

    grade_x_train = grade_x[indices[:test_size]]
    grade_y_train = grade_y[indices[:test_size]]
    grade_x_test = grade_x[indices[test_size:]]
    grade_y_test = grade_y[indices[test_size:]]

    # reg = linear_model.LinearRegression()
    # reg.fit(grade_x_train, grade_y_train)
    # coefs = np.array(reg.coef_)

    # prediction_linear = reg.predict(grade_x_test)
    # print "LR Accuracy (theirs):", accuracy_distance(grade_y_test, prediction_linear)


    # Create and fit a nearest-neighbor classifier
    knn = KNeighborsClassifier()
    knn.fit(grade_x_train, grade_y_train)
    prediction = knn.predict(grade_x_test)
    truth = grade_y_test
    print "KNN Accuracy:", accuracy_score(truth, prediction)
    print "KNN prediction\n", prediction
    print "KNN true values\n", truth


def LRprediction(grade_x, grade_y):
    np.random.seed(1)
    indices = np.random.permutation(len(grade_x))
    test_size = int(round(len(grade_x) * 0.2)) * -1

    grade_x_train = grade_x[indices[:test_size]]
    grade_y_train = grade_y[indices[:test_size]]
    grade_x_test = grade_x[indices[test_size:]]
    grade_y_test = grade_y[indices[test_size:]]

    reg = linear_model.LinearRegression()
    reg.fit(grade_x_train, grade_y_train)
    prediction_linear = reg.predict(grade_x_test)
    prediction_discrete = []
    for i in range(0, len(prediction_linear)):
        result = 0
        if prediction_linear[i] > 0.5:
            result = 1
        prediction_discrete.append(result),

    print "LR Accuracy:", accuracy_score(grade_y_test, prediction_discrete)
    print "LR prediction\n", prediction_discrete
    print "LR true values\n", grade_y_test

    # Create and fit a nearest-neighbor classifier


csvparser = CSVParser()
data_target = csvparser.parse("student-mat.csv")
grade_x = np.array(data_target[0])
grade_y = np.array(data_target[1])


reg2 = linear_model.LinearRegression()
reg2.fit(data_target[0], data_target[1])
coefs_all = np.array(reg2.coef_)

coef_grade_x = []
for i in range(0, len(grade_x)):
    coef_features = []
    for j in range(0, len(grade_x[i])):
        coef_features.append(coefs_all[j] * grade_x[i][j])
    coef_grade_x.append(coef_features)
grade_x_with_weighted_coeffs = np.array(coef_grade_x)

LRprediction(grade_x, grade_y)
KNNprediction(grade_x, grade_y)
KNNprediction(grade_x_with_weighted_coeffs, grade_y)


features = ["school", "sex", "age", "address", "famsize", "pstatus", "medu", "fedu", "traveltime", "studytime",
            "failures",
            "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic", "famrel",
            "freetime", "goout", "dalc", "walc",
            "health", "absences"]


# plt.axis([0,25,-0.1,0.1])
plt.xlabel('Features')
plt.ylabel('Weight')
abscoefs = [abs(x) for x in coefs_all]
for i in range(0, len(coefs_all)):
    if (coefs_all[i] < 0):
        print features[i]

plt.xticks(np.array(range(0, len(features))), features, rotation=90)
plt.plot(np.array(range(0, len(features))), abscoefs, 'r*')
plt.show()

