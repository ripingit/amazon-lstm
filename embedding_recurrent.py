#!/usr/bin/python
#analyzes amazon reviews using a recurrent neural network
#we'll represent words using their pretrained GloVe embeddings

import pickle
import IPython
import numpy as np
from nltk.tokenize.casual import TweetTokenizer as TT
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dropout, Dense, Activation, LSTM, Bidirectional, Masking, Embedding
from sklearn.model_selection import train_test_split
import gc

max_review_length = 370
vector_dimensionality = 300

embedding_matrix = np.load(str(vector_dimensionality) + "d_vocab_vector_matrix.npz")
embedding_matrix = embedding_matrix[embedding_matrix.keys()[0]]


x_unsplit = np.load(str(vector_dimensionality)+ "d_" + str(max_review_length) + "l_indexed_unsplit.npz") 
x_unsplit = x_unsplit[x_unsplit.keys()[0]]

y_file = open("y_unsplit_balanced.txt", "r", encoding = "utf-8")
y_unsplit = np.array([int(line.strip()) for line in y_file])

x_train, x_test, y_train, y_test = train_test_split(x_unsplit, y_unsplit, test_size=0.2)
del x_unsplit



####do the same for a few more models
#the input dimensionality is any number of samples, each containing a sequence of max_review_length 300-d vectors

model = Sequential()
model.add(Embedding(len(embedding_matrix), vector_dimensionality, weights=[embedding_matrix], input_length= max_review_length, trainable=False))
model.add(Masking(mask_value = 0.0, input_shape = (max_review_length, vector_dimensionality)))
model.add(Bidirectional(LSTM(64, activation = "tanh", dropout = 0.2, recurrent_dropout = 0.2)))
model.add(Dense(32, activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(1, activation = "sigmoid"))
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.fit(x_train, y_train, epochs = 20, validation_split = 0.5, batch_size = 340)
score = model.evaluate(x_test, y_test, batch_size = 340)
print(score)

'''
#creates the model
model = Sequential()

#the input dimensionality is any number of samples, each containing a sequence of max_review_length 300-d vectors
model.add(Masking(mask_value = 0, input_shape = (max_review_length, 300)))
model.add(LSTM(64, activation = "tanh", dropout = 0.4, recurrent_dropout = 0.4))
model.add(Dense(32, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation = "sigmoid"))

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.fit(all_train, y_train, epochs = 20, validation_split = 0.5, batch_size = 64)


#test the model
score = model.evaluate(all_test, y_test, batch_size = 128)
print(score)
'''

'''
model = Sequential()
model.add(Masking(mask_value = 0, input_shape = (max_review_length, 300)))
model.add(LSTM(64, activation = "tanh", dropout = 0.2, recurrent_dropout = 0.2))
model.add(Dense(32, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation = "sigmoid"))
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.fit(all_train, y_train, epochs = 20, validation_split = 0.5, batch_size = 64)
score = model.evaluate(all_test, y_test, batch_size = 64)
print(score)
'''
#IPython.embed()
