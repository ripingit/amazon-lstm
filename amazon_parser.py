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

        
x_binary = open("x_files/x_2_way.txt", "w", encoding = "utf-8")
y_binary = open("y_files/y_2_way.txt", "w", encoding = "utf-8")
x_ternary = open("x_files/x_3_way.txt", "w", encoding = "utf-8")
x_quinary = open("x_files/x_5_way.txt", "w", encoding = "utf-8")
y_quinary = open("y_files/y_5_way.txt", "w", encoding = "utf-8")
y_ternary = open("y_files/y_3_way.txt", "w", encoding = "utf-8")

x_binary_balanced = open("x_files/x_2_way_balanced.txt", "w", encoding = "utf-8")
y_binary_balanced = open("y_files/y_2_way_balanced.txt", "w", encoding = "utf-8")
x_ternary_balanced = open("x_files/x_3_way_balanced.txt", "w", encoding = "utf-8")
x_quinary_balanced = open("x_files/x_5_way_balanced.txt", "w", encoding = "utf-8")
y_quinary_balanced = open("y_files/y_5_way_balanced.txt", "w", encoding = "utf-8")
y_ternary_balanced = open("y_files/y_3_way_balanced.txt", "w", encoding = "utf-8")


y_binary_counter = [[0,0],[0,0]] #not,balanced
y_ternary_counter = [[0,0,0],[0,0,0]] #not,balanced 
y_quinary_counter = [[0,0,0,0,0],[0,0,0,0,0]] #not,balanced 
def simplified_parse(formatted_review, simplified_rating):
    if not simplified_rating == 2:
        x_binary.write(formatted_review + "\n")
        y_binary.write(str(simplified_rating)+ "\n")
        y_binary_counter[0][simplified_rating]+=1
        include = random.randint(0,12)
        if simplified_rating == 0:
            x_binary_balanced.write(formatted_review + "\n")
            y_binary_balanced.write(str(simplified_rating)+ "\n")
            y_binary_counter[1][simplified_rating]+=1
        elif include == 0 :
            x_binary_balanced.write(formatted_review + "\n")
            y_binary_balanced.write(str(simplified_rating)+ "\n")
            y_binary_counter[1][simplified_rating]+=1
            
def three_way_parse(formatted_review, simplified_rating):
    x_ternary.write(formatted_review + "\n")
    y_ternary.write(str(simplified_rating)+ "\n")
    y_ternary_counter[0][simplified_rating]+=1
    include_pos = random.randint(0,12)
    include_neutral = random.randint(0,9)
    if simplified_rating == 0:
        x_ternary_balanced.write(formatted_review + "\n")
        y_ternary_balanced.write(str(simplified_rating)+ "\n")
        y_ternary_counter[1][simplified_rating]+=1
    elif (simplified_rating == 1 and include_pos == 0) or (simplified_rating == 2 and include_neutral < 7):
        x_ternary_balanced.write(formatted_review + "\n")
        y_ternary_balanced.write(str(simplified_rating)+ "\n")
        y_ternary_counter[1][simplified_rating]+=1

def unsimplified_parse(formatted_review, unsimplified_rating):
    x_quinary.write(formatted_review + "\n")
    y_quinary.write(str(unsimplified_rating-1)+ "\n")
    y_quinary_counter[0][unsimplified_rating-1]+=1
    include_5 = random.randint(0,24)
    include_4 = random.randint(0,15)
    include_3 = random.randint(0,15)
    include_2 = random.randint(0,3)
    if unsimplified_rating == 1:
        x_quinary_balanced.write(formatted_review + "\n")
        y_quinary_balanced.write(str(unsimplified_rating-1)+ "\n")
        y_quinary_counter[1][unsimplified_rating-1]+=1
    elif (unsimplified_rating  == 2 and include_2 < 3)or(unsimplified_rating  == 3 and include_3 < 5)or(unsimplified_rating  == 4 and include_4 < 2)or(unsimplified_rating  == 5 and include_5 == 0):
        x_quinary_balanced.write(formatted_review + "\n")
        y_quinary_balanced.write(str(unsimplified_rating-1)+ "\n")
        y_quinary_counter[1][unsimplified_rating-1]+=1

for l in parse("reviews_Toys_and_Games_5.json.gz"):
    unsimplified_rating = int(l["overall"])
    simplified_rating = simplify_rating(unsimplified_rating)
    formatted_review = l["reviewText"]
    
    simplified_parse(formatted_review, simplified_rating)
    three_way_parse(formatted_review, simplified_rating)
    unsimplified_parse(formatted_review, unsimplified_rating)
    
print(y_binary_counter)
print(y_ternary_counter)
print(y_quinary_counter)