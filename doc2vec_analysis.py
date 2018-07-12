#takes in training data (reviews) and then makes judgements about a test sample.

import IPython
import random
import gensim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report as class_rep
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression as LR
from nltk.tokenize.casual import TweetTokenizer as TT

size = 150

x_file_name = "x_unsplit_balanced.txt"
y_file_name = "y_unsplit_balanced.txt"
x_file = open(x_file_name, "r", encoding = "utf-8")
y_file = open(y_file_name, "r", encoding = "utf-8")
all_reviews_x_file = open("x_unsplit.txt", "r", encoding = "utf-8")
all_reviews_y_file = open("y_unsplit.txt", "r", encoding = "utf-8")

x_unsplit = x_file.readlines()
y_unsplit = np.array([int(line.strip()) for line in y_file])
all_reviews = all_reviews_x_file.readlines()
all_reviews_y = np.array([int(line.strip()) for line in all_reviews_y_file])

tokenizer = TT(preserve_case = False)
all_reviews_tagged = []
all_reviews_tagged_balanced = []
all_reviews_tagged_balanced_y = []
counter = 0

#turn every review into a tagged document
for line in all_reviews:
    sr = all_reviews_y[counter]
    tl = tokenizer.tokenize(line)
    all_reviews_tagged.append(gensim.models.doc2vec.TaggedDocument(words = tl, tags = [str(counter),str(sr)]))
    include = random.randint(0,12)
    if((sr==0 or (sr == 1 and include == 0))):
        all_reviews_tagged_balanced.append(gensim.models.doc2vec.TaggedDocument(words = tl, tags = [str(counter), str(sr)]))
        all_reviews_tagged_balanced_y.append(sr)
    counter+=1
    

x_train_tagged, x_test_tagged, y_train, y_test = train_test_split(all_reviews_tagged_balanced, all_reviews_tagged_balanced_y, test_size=0.2)
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
np.save("x_train_vectors.npy",train_vecs_dbow)
np.save("x_test_vectors.npy",test_vecs_dbow)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)


'''
#grid search setup
RFC_param_grid = {
    'min_samples_split': [4,6,8,10]
}
'''

#use random forest to classify
print("creating random forest classifier", flush=True)
rfc1 = RFC()

rfc1.fit(train_vecs_dbow, y_train)

p1 = rfc1.predict(test_vecs_dbow)
print(class_rep(y_test, p1))

#use logistic regression to classify
print("creating logistic regression classifier", flush=True)
lr1 = LR()

LR_param_grid = { 'C': [1e-1, 1, 10, 100]}

gs_lr1 = GridSearchCV(estimator = lr1, param_grid = LR_param_grid)

gs_lr1.fit(train_vecs_dbow, y_train)
#rfc1.fit(train_vecs_dbow, y_train)

lrp1 = gs_lr1.predict(test_vecs_dbow)
#lrp1 = rfc1.predict(test_vecs_dbow)
print(class_rep(y_test, lrp1))
print(gs_lr1.best_params_)

IPython.embed()