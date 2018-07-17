#!/usr/bin/python
#analyzes amazon reviews using a recurrent neural network
#we'll represent words using their pretrained GloVe embeddings

import pickle
#import IPython
import numpy as np
from nltk.tokenize.casual import TweetTokenizer as TT
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dropout, Dense, Activation, LSTM, Flatten, Masking
from sklearn.model_selection import train_test_split
import gc

'''
#load the word vectors from GloVe 
def loadGloveModel(gloveFile):
    print ("Loading GloVe Model")
    f = open(gloveFile,'r', encoding = "utf-8")
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model),"words loaded!")
    return model

glove_model = loadGloveModel("../glove.42B.300d.txt")
'''
with open("review_vocab_2.pickle", 'rb') as handle:
    glove_model = pickle.load(handle)

vector_dimensionality = 300

#load the reviews    
x_file = open("x_unsplit_balanced.txt", "r", encoding = "utf-8")
y_file = open("y_unsplit_balanced.txt", "r", encoding = "utf-8")

#split the data into train and test
x_unsplit = x_file.readlines()
y_unsplit = np.array([int(line.strip()) for line in y_file])

x_train, x_test, y_train, y_test = train_test_split(x_unsplit, y_unsplit, test_size=0.2)

train_len = len(x_train)
test_len =  len(x_test)

#convert reviews into arrays of word vectors
max_review_length = 320
tokenizer = TT(preserve_case = False)

#returns a 3D matrix representing the (sample, timestep, feature) of a GloVe-translated review 
#each matrix only has one sample, so really this a 
#the # of timesteps = max_review_length (number of words), # of features = vector dimensionality (for these GloVe vectors), here 300

def get_all_glove_reviews(data):
    all_reviews_matrix = np.empty((len(data), max_review_length, vector_dimensionality))
    review_index = 0
    for line in data:
        review_matrix = np.empty((max_review_length, vector_dimensionality))
        index = 0
        processed_line = tokenizer.tokenize(line)
        for word in processed_line:
            if(word in glove_model and index < max_review_length):
                review_matrix[index] = glove_model[word]
            elif (index < max_review_length):
                review_matrix[index] = np.zeros(vector_dimensionality)
            index += 1
        if(index < max_review_length):
            for padding_index in range(index, max_review_length):
                review_matrix[padding_index] = np.zeros(vector_dimensionality)
        #note that the triple is just a tuple wrapped in another matrix 
        #this is done to conform with the expected input shape of all Keras RNN layers
        all_reviews_matrix[review_index] = review_matrix
        review_index += 1
    return all_reviews_matrix
        
#get the training and testing datasets
all_train = get_all_glove_reviews(x_train)
all_test = get_all_glove_reviews(x_test)

#explicitly eliminate the glove_model, as it is no longer needed
del glove_model
gc.collect()

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
####do the same for a few more models
#the input dimensionality is any number of samples, each containing a sequence of max_review_length 300-d vectors

model = Sequential()
model.add(Masking(mask_value = 0.0, input_shape = (max_review_length, vector_dimensionality)))
model.add(LSTM(64, activation = "tanh", dropout = 0.2, recurrent_dropout = 0.2))
model.add(Dense(32, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation = "sigmoid"))
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.fit(all_train, y_train, epochs = 13, validation_split = 0.5, batch_size = 128)
score = model.evaluate(all_test, y_test, batch_size = 128)
print(score)
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
