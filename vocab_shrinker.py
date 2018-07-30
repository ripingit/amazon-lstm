#converts the large list of GloVe vectors into one that contains only the necessary ones

import IPython
import numpy as np
#from nltk.tokenize.casual import TweetTokenizer as TT
from keras.preprocessing.text import text_to_word_sequence
import gc
from sklearn.feature_extraction.text import CountVectorizer
import collections
import pickle
import argparse

parser = argparse.ArgumentParser(description='Specify the dimensionality of the vectors')
parser.add_argument("-d", action = "store", nargs = '?', default = 300, type = int, dest = "vector_dimensionality", help = "the dimensionality of the GloVe vectors used")
parser.add_argument("-s", action = "store", nargs = '?', default = 2, type = int, dest = "simplification", help = "2: binary (default); 3: negative, neutral, positive; 5: 1-5 stars")
args = parser.parse_args()

vector_dimensionality = args.vector_dimensionality
simplification_level = args.simplification
simp_string = "2" if simplification_level == 2 else "multi"

#tokenizer = TT(preserve_case = False)

#load the reviews    
x_file = open("x_" +simp_string+"_way_balanced.txt", "r", encoding = "utf-8")
x_unsplit = x_file.readlines()
x_unsplit_tokenized = [text_to_word_sequence(review) for review in x_unsplit]
#x_unsplit_tokenized = [tokenizer.tokenize(review) for review in x_unsplit]

del x_unsplit
gc.collect()

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
    f.close()
    return model

glove_model = loadGloveModel("../glove.6B/glove.6B." + str(vector_dimensionality) + "d.txt")

counter = collections.Counter()
review_vocab = {}

for review in x_unsplit_tokenized:
    for token in review:
        counter[token] += 1
            
vocab_size = 12000

for word, index in counter.most_common(vocab_size):
    if word in glove_model:
        review_vocab[word] = glove_model[word]
        
print("The size of the review vocabulary is: %s" % len(review_vocab))
            
with open(simp_string + "_way_" +str(vector_dimensionality) + 'd_review_vocab_4.pickle', 'wb') as handle:
    pickle.dump(review_vocab, handle, protocol=4)
    