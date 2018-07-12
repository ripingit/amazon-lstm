#takes as input amazon review files in python dictionary format from
#outputs:
#1. all reviews, separated by line
#2. all ratings, separated by line, aligned with reviews, simplified to negative (reviews scoring less than 4) and positive (all others)
#3. all ratings, separated by line, algined with reviews, unsimplified

import gzip
import random
from langdetect import detect

#turns 1 or 2 ratings into 0, 4 to 5 into 1 (negative or positive)
def simplify_rating(rating):
    if(rating < 3):
        return 0
    elif(rating == 3):
        return 2
    else:
        return 1

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)
    g.close()
'''    
def format_review(text):
    output = text.lower()
    return ST.tokenize(output)
'''
        
x_unsplit = open("x_unsplit.txt", "w", encoding = "utf-8")
y_unsplit = open("y_unsplit.txt", "w", encoding = "utf-8")
y_unsplit_unsimplified = open("y_unsplit_unsimplified.txt", "w", encoding = "utf-8")

x_unsplit_balanced = open("x_unsplit_balanced.txt", "w", encoding = "utf-8")
y_unsplit_balanced = open("y_unsplit_balanced.txt", "w", encoding = "utf-8")
y_unsplit_unsimplified_balanced = open("y_unsplit_unsimplified_balanced.txt", "w", encoding = "utf-8")

counter = 0
bins = [0,0,0,0,0]
s_bins = [0,0]

counter_balanced = 0
bins_balanced = [0,0,0,0,0]
s_bins_balanced = [0,0]

for l in parse("reviews_Toys_and_Games_5.json.gz"):

    sr = simplify_rating(int(l["overall"]))
    formatted_review = l["reviewText"]
    
    if(sr != 2):
        x_unsplit.write(formatted_review + "\n")
        y_unsplit.write(str(sr)+ "\n")
        y_unsplit_unsimplified.write(str(int(l["overall"]))+ "\n")
        bins[int(l["overall"]) - 1] += 1
        s_bins[sr] += 1
        counter += 1

    include = random.randint(0,12)
    if((sr==0 or (sr == 1 and include == 0))):
        x_unsplit_balanced.write(formatted_review + "\n")
        y_unsplit_balanced.write(str(sr)+ "\n")
        y_unsplit_unsimplified_balanced.write(str(int(l["overall"]))+ "\n")
        bins_balanced[int(l["overall"]) - 1] += 1
        s_bins_balanced[sr] += 1
        counter_balanced += 1
    
print("parsed %s entries total" %counter)
print(bins)
print(s_bins)

print("parsed %s balanced entries" %counter_balanced)
print(bins_balanced)
print(s_bins_balanced)
'''
oversample_ratio = 1

while oversample_ratio > 0:
    for l in parse("reviews_Grocery_and_Gourmet_Food_5.json.gz"):
        if(not simplify_rating(int(l["overall"]))):
            x_unsplit.write(l["reviewText"].lower() + "\n")
            y_unsplit.write(str(simplify_rating(int(l["overall"])))+ "\n")
            y_unsplit_unsimplified.write(str(int(l["overall"]))+ "\n")
    oversample_ratio -=1
            
'''