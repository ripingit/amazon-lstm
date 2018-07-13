#converts the large list of GloVe vectors into one that contains only the necessary ones

import IPython
import numpy as np
from nltk.tokenize.casual import TweetTokenizer as TT
import gc
from sklearn.feature_extraction.text import CountVectorizer
import collections
import pickle

tokenizer = TT(preserve_case = False)

#load the reviews    
x_file = open("x_unsplit_balanced.txt", "r", encoding = "utf-8")
x_unsplit = x_file.readlines()
x_unsplit_tokenized = [tokenizer.tokenize(review) for review in x_unsplit]

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

glove_model = loadGloveModel("../glove.42B.300d.txt")

counter = collections.Counter()
review_vocab = {}

for review in x_unsplit_tokenized:
    for token in review:
        counter[token] += 1
            
vocab_size = 17000

for word in counter.most_common(vocab_size):
    if word in glove_model:
        review_vocab[word] = glove_model[word]
        
print("The size of the review vocabulary is: %s" % len(review_vocab))
            
with open('review_vocab_2.pickle', 'wb') as handle:
    pickle.dump(review_vocab, handle, protocol=2)
    
IPython.embed()