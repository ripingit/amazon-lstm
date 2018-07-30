#takes as input amazon review files in python dictionary format from
#outputs:
#1. all reviews, separated by line
#2. all ratings, separated by line, aligned with reviews, simplified to negative (reviews scoring less than 4) and positive (all others)
#3. all ratings, separated by line, algined with reviews, unsimplified

import gzip
import random

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

        
x_binary = open("x_2_way.txt", "w", encoding = "utf-8")
y_binary = open("y_2_way.txt", "w", encoding = "utf-8")
x_multi = open("x_multi_way.txt", "w", encoding = "utf-8")
y_quinary = open("y_5_way.txt", "w", encoding = "utf-8")
y_ternary = open("y_3_way.txt", "w", encoding = "utf-8")

x_binary_balanced = open("x_2_way_balanced.txt", "w", encoding = "utf-8")
y_binary_balanced = open("y_2_way_balanced.txt", "w", encoding = "utf-8")
x_multi_balanced = open("x_multi_way_balanced.txt", "w", encoding = "utf-8")
y_quinary_balanced = open("y_5_way_balanced.txt", "w", encoding = "utf-8")
y_ternary_balanced = open("y_3_way_balanced.txt", "w", encoding = "utf-8")

counter = 0
bins = [0,0,0,0,0]
s_bins = [0,0]

counter_balanced = 0
bins_balanced = [0,0,0,0,0]
s_bins_balanced = [0,0]

def simplified_parse(formatted_review, simplified_rating, include):
    if not simplified_rating == 2:
        x_binary.write(formatted_review + "\n")
        y_binary.write(str(simplified_rating)+ "\n")
        if simplified_rating == 0:
            x_binary_balanced.write(formatted_review + "\n")
            y_binary_balanced.write(str(simplified_rating)+ "\n")
        elif include == 0 :
            x_binary_balanced.write(formatted_review + "\n")
            y_binary_balanced.write(str(simplified_rating)+ "\n")
            
            
def three_way_parse(formatted_review, simplified_rating, include):
    x_multi.write(formatted_review + "\n")
    y_ternary.write(str(simplified_rating)+ "\n")
    if not simplified_rating == 1:
        x_multi_balanced.write(formatted_review + "\n")
        y_ternary_balanced.write(str(simplified_rating)+ "\n")
    elif include == 0 :
        x_multi_balanced.write(formatted_review + "\n")
        y_ternary_balanced.write(str(simplified_rating)+ "\n")

def unsimplified_parse(formatted_review, unsimplified_rating, include):
    y_quinary.write(str(unsimplified_rating)+ "\n")
    if not (unsimplified_rating == 4 or unsimplified_rating == 5):
        y_quinary_balanced.write(str(unsimplified_rating)+ "\n")
    elif include == 0 :
        y_quinary_balanced.write(str(unsimplified_rating)+ "\n")

for l in parse("reviews_Toys_and_Games_5.json.gz"):
    unsimplified_rating = int(l["overall"])
    simplified_rating = simplify_rating(unsimplified_rating)
    formatted_review = l["reviewText"]
    include = random.randint(0,12)
    
    simplified_parse(formatted_review, simplified_rating, include)
    three_way_parse(formatted_review, simplified_rating, include)
    unsimplified_parse(formatted_review, unsimplified_rating, include)
    
'''
print("parsed %s entries total" %counter)
print(bins)
print(s_bins)

print("parsed %s balanced entries" %counter_balanced)
print(bins_balanced)
print(s_bins_balanced)
'''