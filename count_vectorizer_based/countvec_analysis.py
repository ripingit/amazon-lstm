#analyzes the reviews using a count-based method instead of doc2vec
#the feature extractor used is scikit learn's count vectorizer

import IPython
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report as class_rep
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse 

parser = argparse.ArgumentParser(description='Specify the degree of class simplification you want')
parser.add_argument("-s", action = "store", nargs = '?', default = 2, type = int, dest = "simplification", help = "2: binary; 3: negative, neutral, positive; 5: 1-5 stars")
parser.add_argument("-t", action = "store", nargs = '?', default = 0, type = int, dest = "tfidf", help = "1 to use tf-idf vectors, 0 to use plain counts")
args = parser.parse_args()

simplification = args.simplification
tf_idf = args.tfidf

x_file_name = "../x_" + str(simplification) +"_way_balanced.txt"
y_file_name_dict  = {2:"../y_2_way_balanced.txt",
                     3:"../y_3_way_balanced.txt",
                     5:"../y_5_way_balanced.txt"}
y_file_name = y_file_name_dict[simplification]


x_file = open(x_file_name, "r", encoding = "utf-8")
y_file = open(y_file_name, "r", encoding = "utf-8")

x_unsplit = x_file.readlines()
y_unsplit = np.array([int(line.strip()) for line in y_file])

x_train, x_test, y_train, y_test = train_test_split(x_unsplit, y_unsplit, test_size=0.2)

cv = CountVectorizer(binary = 1, ngram_range = (1,2)) if tf_idf == 0 else TfidfVectorizer(binary = 1, ngram_range = (1,2))


cv.fit(x_unsplit)

x_train_matrix = cv.transform(x_train)
x_test_matrix = cv.transform(x_test)

np.save("x_train_cv.npy",x_train_matrix)
np.save("x_test_cv.npy",x_test_matrix)
np.save("y_train_cv.npy", y_train)
np.save("y_test_cv.npy", y_test)

mnb1 = MNB()
lgr1 = LR(multi_class = "ovr" if simplification == 2 else "multinomial", solver = "liblinear" if simplification == 2 else "newton-cg", max_iter = 200)
rmf1 = RFC()
#svm1 = SVC()

mnb_param_distributions = {'alpha': [1]}
lgr_param_distributions = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100]}
rmf_param_distributions = {'max_depth': [20,40,60,80,100,None], 'min_samples_leaf': [1,2,4],'min_impurity_decrease': [0,1,2]}
#svm_param_distributions = {}

rs_mnb = RandomizedSearchCV(estimator = mnb1, param_distributions = mnb_param_distributions, n_iter = 1)
rs_lgr = RandomizedSearchCV(estimator = lgr1, param_distributions = lgr_param_distributions, n_iter = 5)
rs_rmf = RandomizedSearchCV(estimator = rmf1, param_distributions = rmf_param_distributions, n_iter = 5)
#rs_svm = RandomizedSearchCV(estimator = svm1, param_distributions = svm_param_distributions)


def print_results(classifier):
    predictions = classifier.predict(x_test_matrix)
    print(class_rep(y_test, predictions))
    print("The overall score (accuracy) was: " + str(classifier.score(x_test_matrix, y_test)))
    print(classifier.best_params_)
    return predictions
    

rs_mnb.fit(x_train_matrix, y_train)
predictions_lgr = print_results(rs_mnb)
    
rs_lgr.fit(x_train_matrix, y_train)
predictions_lgr = print_results(rs_lgr)

rs_rmf.fit(x_train_matrix, y_train)
predictions_lgr = print_results(rs_rmf)
'''
rs_svm.fit(x_train_matrix, y_train)
predictions_lgr = print_results(rs_svm)
'''
#IPython.embed()