#converts the large list of GloVe vectors into one that contains only the necessary ones

import IPython
import numpy as np
from nltk.tokenize.casual import TweetTokenizer as TT
import gc
from sklearn.feature_extraction.text import CountVectorizer

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
    return model

glove_model = loadGloveModel("../glove.42B.300d.txt")

new_dict = {}
for review in x_unsplit_tokenized:
    for token in review:
        if (token in glove_model):
            new_dict[token] = glove_model[token]
            del glove_model[token]
            
            
IPython.embed()