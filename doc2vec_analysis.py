#takes in training data (reviews) and then makes judgements about a test sample.

import IPython
import random
import gensim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report as class_rep
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.linear_model import LogisticRegression as LR
from nltk.tokenize.casual import TweetTokenizer as TT
import argparse

size = 150

parser = argparse.ArgumentParser(description='Specify the degree of class simplification you want')
parser.add_argument("-s", action = "store", nargs = '?', default = 2, type = int, dest = "simplification", help = "2: binary; 3: negative, neutral, positive; 5: 1-5 stars")
args = parser.parse_args()

simplification = args.simplification

x_file_name = "x_files/x_" + str(simplification) +"_way_balanced.txt"
y_file_name = "y_files/y_" + str(simplification) +"_way_balanced.txt"
x_file = open(x_file_name, "r", encoding = "utf-8")
y_file = open(y_file_name, "r", encoding = "utf-8")
all_reviews_x_file = open("x_files/x_5_way.txt", "r", encoding = "utf-8")
all_reviews_y_file = open("y_files/y_5_way.txt", "r", encoding = "utf-8")

x_unsplit = x_file.readlines()
y_unsplit = np.array([int(line.strip()) for line in y_file])
all_reviews = all_reviews_x_file.readlines()
all_reviews_y = np.array([int(line.strip()) for line in all_reviews_y_file])

tokenizer = TT(preserve_case = False)
all_reviews_tagged = []
x_unsplit_tagged = []
counter = 0

#turn every review into a tagged document
for review in all_reviews:
    #sr = all_reviews_y[counter]
    tl = tokenizer.tokenize(review)
    all_reviews_tagged.append(gensim.models.doc2vec.TaggedDocument(words = tl, tags = [str(counter)]))
    counter+=1
    #include = random.randint(0,12)
    #if((sr==0 or (sr == 1 and include == 0))):
    #    all_reviews_tagged_balanced.append(gensim.models.doc2vec.TaggedDocument(words = tl, tags = [str(counter), str(sr)]))
    #    all_reviews_tagged_balanced_y.append(rating)
    #counter+=1
    
for review in x_unsplit:
    tl = tokenizer.tokenize(review)
    x_unsplit_tagged.append(gensim.models.doc2vec.TaggedDocument(words = tl, tags = [str(counter)]))
    counter+=1
    
x_train_tagged, x_test_tagged, y_train, y_test = train_test_split(x_unsplit_tagged, y_unsplit, test_size=0.2)
print("size of x_train is %s, size of x_test is %s" %(len(x_train_tagged), len(x_test_tagged)))

model_dbow = gensim.models.Doc2Vec(min_count = 10, window = 10, vector_size = size, sample = 1e-5, negative = 5, workers = 3, iter = 8)
model2 = gensim.models.Doc2Vec(min_count = 10, window = 10, vector_size = size, sample = 1e-5, negative = 5, workers = 3, iter = 8)
model3 = gensim.models.Doc2Vec(min_count = 10, window = 10, vector_size = size, sample = 1e-5, negative = 5, workers = 3, iter = 8)

#build the vocab over the entirety of the reviews
model_dbow.build_vocab(all_reviews_tagged)

print("beginning training on training_reviews", flush=True)
#but only train with the training reviews
model_dbow.train(all_reviews_tagged, epochs = model_dbow.iter, total_examples=model_dbow.corpus_count)

#infer vectors
print("inferring vectors", flush=True)
train_vecs_dbow = [model_dbow.infer_vector(doc[0]) for doc in x_train_tagged]
test_vecs_dbow = [model_dbow.infer_vector(doc[0]) for doc in x_test_tagged]

#save the train and test vectors
np.save("x_files/x_train_vectors.npy",train_vecs_dbow)
np.save("x_files/x_test_vectors.npy",test_vecs_dbow)
np.save("y_files/y_train.npy", y_train)
np.save("y_files/y_test.npy", y_test)

#mnb1 = MNB()
lgr1 = LR(multi_class = "ovr" if simplification == 2 else "multinomial", solver = "liblinear" if simplification == 2 else "saga", max_iter = 200)
rmf1 = RFC()
#svm1 = SVC()

#mnb_param_distributions = {'alpha': [0,0.5,1]}
lgr_param_distributions = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100]}
rmf_param_distributions = {'max_depth': [20,40,60,80,100,None], 'min_samples_leaf': [1,2,4],'min_impurity_decrease': [0,1,2]}
#svm_param_distributions = {}

#rs_mnb = RandomizedSearchCV(estimator = mnb1, param_distributions = mnb_param_distributions, n_iter = 3)
rs_lgr = RandomizedSearchCV(estimator = lgr1, param_distributions = lgr_param_distributions, n_iter = 5)
rs_rmf = RandomizedSearchCV(estimator = rmf1, param_distributions = rmf_param_distributions, n_iter = 15)
#rs_svm = RandomizedSearchCV(estimator = svm1, param_distributions = svm_param_distributions)


def print_results(classifier):
    predictions = classifier.predict(test_vecs_dbow)
    print(class_rep(y_test, predictions))
    print("The overall score (accuracy) was: " + str(classifier.score(test_vecs_dbow, y_test)))
    print(classifier.best_params_)
    return predictions
    

#rs_mnb.fit(train_vecs_dbow, y_train)
#predictions_lgr = print_results(rs_mnb)
    
rs_lgr.fit(train_vecs_dbow, y_train)
predictions_lgr = print_results(rs_lgr)

rs_rmf.fit(train_vecs_dbow, y_train)
predictions_lgr = print_results(rs_rmf)
'''
rs_svm.fit(train_vecs_dbow, y_train)
predictions_lgr = print_results(rs_svm)
'''

IPython.embed()