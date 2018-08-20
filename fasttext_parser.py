#!/usr/bin/env python3

#parses the given amazon reviews in such a way that they are readable by fasttext

import gzip
import random
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(description='Specify the level of simplification')
parser.add_argument("-s", action = "store", nargs = '?', default = 2, type = int, dest = "simplification", help = "2: binary (default); 3: negative, neutral, positive; 5: 1-5 stars")
parser.add_argument("-c", action = "store", nargs = '?', default = "Toys", type = string, dest = "category", help = "what category to classify (should be same as folder name)")
args = parser.parse_args()

simplification_level = args.simplification
category = args.category

x_unbalanced = open(category+"/x_files/x_"+str(simplification_level)+"_way.txt", "r", encoding = "utf-8")
y_unbalanced = open(category+"/y_files/y_"+str(simplification_level)+"_way.txt", "r", encoding = "utf-8")
x_balanced = open(category+"/x_files/x_"+str(simplification_level)+"_way_balanced.txt", "r", encoding = "utf-8")
y_balanced = open(category+"/y_files/y_"+str(simplification_level)+"_way_balanced.txt", "r", encoding = "utf-8")

ft_train_ub = open(category+"/fasttext_files/ft_"+str(simplification_level)+"_way_train_ub.txt", "w", encoding = "utf-8")
ft_test_ub = open(category+"/fasttext_files/ft_"+str(simplification_level)+"_way_test_ub.txt", "w", encoding = "utf-8")
ft_train = open(category+"/fasttext_files/ft_"+str(simplification_level)+"_way_train.txt", "w", encoding = "utf-8")
ft_test = open(category+"/fasttext_files/ft_"+str(simplification_level)+"_way_test.txt", "w", encoding = "utf-8")

unbalanced_reviews = ["__label__"+str(ub_score).strip()+ " " +ub_review for ub_review, ub_score in zip(x_unbalanced, y_unbalanced)]
reviews = ["__label__"+str(score).strip()+" "+review for review, score in zip(x_balanced, y_balanced)]

x_train, x_test = train_test_split(reviews, test_size=0.2)
x_train_ub, x_test_ub = train_test_split(unbalanced_reviews, test_size=0.2)

for line in x_train:
    ft_train.write(line)
    
for line in x_test:
    ft_test.write(line)
    
for line in x_train_ub:
    ft_train_ub.write(line)
    
for line in x_test_ub:
    ft_test_ub.write(line)