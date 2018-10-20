# Predicting positivity of the sentiment of Amazon Reviews
This is a project built during the summer of 2018 while I was at the Universitat Pompeu Fabra. It's fairly simple, and aims only to predict the rating of a given review text. 
It can predict whether the text was positive / negative (5 or 1 stars), positive / neutral / negative (5/3/1 stars), or it can try to recover the original, 1-5 star rating of the text.

Although what is available in this repository is consists of only one deep learning model, I attempted this same task with a variety of deep learning and machine learning models. 
It is my intention to upload a short paper detailing the outcomes and relative performance of these various models in the near future.

## Getting Started
The Python scripts contained in this repository were written in Python 3. 
The model itself was built in Keras (2.2.0), running with Tensorflow (1.10) as the backend.

### Downloading the datasets
The Amazon review data can be downloaded here: http://jmcauley.ucsd.edu/data/amazon/

This model uses pretrained GloVe word embeddings. These can be found here: https://nlp.stanford.edu/projects/glove/ . I used the 42B Common Crawl embeddings.

## Running the model

### Data Preprocessing
This repository contains some scripts to help preprocess the data. These include amazon_parser, vocab_shrinker, and get_word_mapping.

amazon_parser extracts the reviews and ratings for each review, and creates a balanced dataset by undersampling overrepresented classes.

vocab_shrinker creates a vocabulary words that both are part of the top *n* words in the reviews that it is run on, and exist in the GloVe dataset.

get_word_mapping creates a mapping between indices and words that will be used as the initial weights for the word embeddings. 
It also transforms the corresponding set of amazon reviews into a matrix of numbers that will be fed into the deep learning model as data (and interpreted as words via the embedding matrix).

Each of these three should be run, in the above order. They are run with the following options: The tag {[a],[v],[g]} shows which script takes which options.

-c: [a][v][g] the directory in which the reviews are found. This is the category of the reviews (download the reviews by category from the aforegiven link)

-d: [v][g] the dimensionality of the vectors being used (in case you want to use vectors other than 300d)

-s: [v][g] the degree of simplification you want; 2: binary classifiction; 3: ternary classification; 5: 1-5 star classification

-r: [g] review length; the length to which reviews will be padded / cut. You can trade-off between memory and accuracy this way.

### Running the model

Now that you've run these three scripts, you'll have generated the necessary training files to run the model. 
Simply run embedding_recurrent.py with all of the above arguments, plus another, "-e", for the number of epochs to train for.
