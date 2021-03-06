#!/usr/bin/python
#analyzes amazon reviews using a recurrent neural network
#we'll represent words using their pretrained GloVe embeddings

import pickle
import IPython
import numpy as np
from nltk.tokenize.casual import TweetTokenizer as TT
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dropout, Dense, Activation, Embedding, Flatten
from sklearn.model_selection import train_test_split
import gc
import argparse 

parser = argparse.ArgumentParser(description='Specify the review length, vector dimensionality, and number of classes that you want.')
parser.add_argument("-s", action = "store", nargs = '?', default = 2, type = int, dest = "simplification", help = "2: binary (default); 3: negative, neutral, positive; 5: 1-5 stars")
parser.add_argument("-d", action = "store", nargs = '?', default = 300, type = int, dest = "dimensionality", help = "specify vector dimensionality ")
parser.add_argument("-r", action = "store", nargs = '?', default = 370, type = int, dest = "review_length", help = "max_review_length (generate reviews of this length using get_word_mapping)")
parser.add_argument("-e", action = "store", nargs = '?', default = 30, type = int, dest = "epochs", help = "number of epochs")
args = parser.parse_args()


max_review_length = args.review_length
vector_dimensionality = args.dimensionality
simplification_level = args.simplification
num_epochs = args.epochs

embedding_matrix = np.load("neural_train/"+str(simplification_level) + "_way_" +str(vector_dimensionality) + "d_vocab_vector_matrix.npz")
embedding_matrix = embedding_matrix[embedding_matrix.keys()[0]]


x_unsplit = np.load("neural_train/"+str(simplification_level) + "_way_" +str(vector_dimensionality)+ "d_" + str(max_review_length) + "l_indexed_unsplit.npz") 
x_unsplit = x_unsplit[x_unsplit.keys()[0]]

y_file = open("y_files/y_"+str(simplification_level) + "_way_balanced.txt", "r", encoding = "utf-8")
y_unsplit = np.array([int(line.strip()) for line in y_file])


y_unsplit = y_unsplit if simplification_level == 2 else to_categorical(y_unsplit)


x_train, x_test, y_train, y_test = train_test_split(x_unsplit, y_unsplit, test_size=0.2)
del x_unsplit

#print("simplification level is: " + str(y_unsplit.shape))

loss_type = "binary_crossentropy" if simplification_level == 2 else "categorical_crossentropy"
number_of_units = 1 if simplification_level == 2 else simplification_level

####do the same for a few more models
#the input dimensionality is any number of samples, each containing a sequence of max_review_length 300-d vectors

model = Sequential()
model.add(Embedding(len(embedding_matrix), vector_dimensionality, input_length= max_review_length, trainable=True))
model.add(Flatten())
model.add(Dense(120, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(50, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(number_of_units, activation = "sigmoid"))
model.compile(optimizer = "adam", loss = loss_type, metrics = ["accuracy"])
model.fit(x_train, y_train, epochs = num_epochs, validation_split = 0.2, batch_size = 100)
score = model.evaluate(x_test, y_test, batch_size = 200)
print(score)

#IPython.embed()
