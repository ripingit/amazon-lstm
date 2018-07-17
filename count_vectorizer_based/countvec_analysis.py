#analyzes the reviews using a count-based method instead of doc2vec
#the feature extractor used is scikit learn's count vectorizer

import IPython
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report as class_rep
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_extraction.text import CountVectorizer

x_file_name = "x_unsplit_balanced.txt" #input("Enter the name of the unsplit file for x values: ")
y_file_name = "y_unsplit_balanced.txt" #input("Enter the name of the unsplit file for y values: ")


x_file = open(x_file_name, "r", encoding = "utf-8")
y_file = open(y_file_name, "r", encoding = "utf-8")

x_unsplit = x_file.readlines()
y_unsplit = np.array([int(line.strip()) for line in y_file])

x_train, x_test, y_train, y_test = train_test_split(x_unsplit, y_unsplit, test_size=0.2)

cv = CountVectorizer()

cv.fit(x_unsplit)

x_train_matrix = cv.transform(x_train)
x_test_matrix = cv.transform(x_test)

np.save("x_train_cv.npy",x_train_matrix)
np.save("x_test_cv.npy",x_test_matrix)
np.save("y_train_cv.npy", y_train)
np.save("y_test_cv.npy", y_test)

print("creating logistic regression classifier", flush=True)
lr1 = LR()

LR_param_grid = { 'C': [1e-1, 1, 10, 100]}

gs_lr1 = GridSearchCV(estimator = lr1, param_grid = LR_param_grid)

gs_lr1.fit(x_train_matrix, y_train)
#rfc1.fit(train_vecs_dbow, y_train)

lrp1 = gs_lr1.predict(x_test_matrix)
#lrp1 = rfc1.predict(test_vecs_dbow)
print(class_rep(y_test, lrp1))
print(gs_lr1.best_params_)



IPython.embed()