#analyzes amazon reviews using a recurrent neural network
#we'll represent words using the default Keras tokenizer

import IPython
import numpy as np
from nltk.tokenize.casual import TweetTokenizer as TT
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dropout, Dense, Activation, LSTM, Flatten, Masking
from sklearn.model_selection import train_test_split
import gc

#load the reviews    
x_file = open("x_unsplit_balanced.txt", "r", encoding = "utf-8")
y_file = open("y_unsplit_balanced.txt", "r", encoding = "utf-8")

#split the data into train and test
x_unsplit = [for line in x_file]
y_unsplit = np.array([int(line.strip()) for line in y_file])

x_train, x_test, y_train, y_test = train_test_split(x_unsplit, y_unsplit, test_size=0.2)
