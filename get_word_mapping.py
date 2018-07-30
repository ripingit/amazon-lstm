#!/usr/bin/python
#analyzes amazon reviews using a recurrent neural network
#we'll represent words using their pretrained GloVe embeddings

import pickle
import IPython
import numpy as np
import tensorflow as tf
#from nltk.tokenize.casual import TweetTokenizer as TT
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import gc
import argparse 

parser = argparse.ArgumentParser(description='Specify the dimensionality of the vectors and length of reviews')
parser.add_argument("-d", action = "store", nargs = '?', default = 300, type = int, dest = "vector_dimensionality", help = "the dimensionality of the GloVe vectors used")
parser.add_argument("-r", action = "store", nargs = '?', default = 370, type = int, dest = "review_length", help = "the review length")
parser.add_argument("-s", action = "store", nargs = '?', default = 2, type = int, dest = "simplification", help = "2: binary (default); 3: negative, neutral, positive; 5: 1-5 stars")
args = parser.parse_args()


vector_dimensionality = args.vector_dimensionality
review_length = args.review_length
simplification_level = args.simplification
simp_string = "2" if simplification_level == 2 else "multi"

with open(simp_string + "_way_" + str(vector_dimensionality) + "d_review_vocab_4.pickle", 'rb') as handle:
    glove_model = pickle.load(handle)

#tt = TT(preserve_case = False)

#load the reviews    
x_file = open("x_" +simp_string + "_way_unsplit_balanced.txt", "r", encoding = "utf-8")


#split the data into train and test
x_unsplit = x_file.readlines()
x_unsplit_tokenized = [text_to_word_sequence(review) for review in x_unsplit]


#convert reviews into arrays of word vectors
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_unsplit_tokenized)
vocab_length = len(tokenizer.word_index)
glove_vocab_length = len(glove_model)
vector_dimensionality = len(glove_model["the"])

vocab_array = np.zeros((glove_vocab_length, vector_dimensionality))
for word, index in tokenizer.word_index.items():
    if word in glove_model and index < glove_vocab_length :
        vocab_array[index] = glove_model[word]

np.savez_compressed(simp_string + "_way_" +str(vector_dimensionality) + "d_vocab_vector_matrix.npz", vocab_array)
del vocab_array

#returns a 3D matrix representing the (sample, timestep, feature) of a GloVe-translated review 
#each matrix only has one sample, so really this a 
#the # of timesteps = max_review_length (number of words), # of features = vector dimensionality (for these GloVe vectors), here 300

wi = tokenizer.word_index

def get_reviews(data):
    num_reviews = len(data)
    all_reviews_matrix = np.empty((num_reviews, review_length))
    for review in range(num_reviews):
        #review_matrix = np.empty((review_length))
        this_text_review = data[review]
        this_text_review_length = len(this_text_review)
        for number, word_counter in zip(this_text_review, range(this_text_review_length)):
            if(word_counter < review_length and number < glove_vocab_length):
                all_reviews_matrix[review][word_counter] = number
            elif (word_counter < review_length):
                all_reviews_matrix[review][word_counter] = 0
        if(word_counter < review_length):
            for padding_index in range(word_counter + 1, review_length):
                all_reviews_matrix[review][padding_index] = 0
        #note that the triple is just a tuple wrapped in another matrix 
        #this is done to conform with the expected input shape of all Keras RNN layers
        #all_reviews_matrix[review] = review_matrix
    return all_reviews_matrix


#get the training and testing datasets
all_unsplit = tokenizer.texts_to_sequences(x_unsplit_tokenized)
np_unsplit = get_reviews(all_unsplit)
np.savez_compressed(simp_string + "_way_" +str(vector_dimensionality) + "d_" + str(review_length) + "l_indexed_unsplit.npz", np_unsplit)
