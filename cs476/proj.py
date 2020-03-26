# proj.py - tokenization
#
# Author: Daniel Choo
# Date:   02/03/20

import os
import re
import sys
import time
import json
import math
import nltk
import string
import stoplist
import progressbar
import numpy as np
from bs4 import BeautifulSoup as Soup
from collections import Counter


# Instantiating cached global variable
stopset = stoplist.stopwords()


def write(path, contents):
    """ write(path, contents): 
        takes in a path and contents, and writes/dumps the file as a JSON.

        Returns:    Nothing [None]
    """
    try:    
        with open(path, "w+", encoding="utf-8") as file:
            file.write(json.dumps(contents, ensure_ascii=False, indent=2))

    except Exception:
        raise RuntimeError("There was something wrong with writing the file.")

def stop(var):
    """ stop(var):
        takes in a word and checks whether or not it is in stopset.

        Return(s): True/False [Bool]
    """
    if var in stopset:
        return True
    else:
        return False

def repeating(var):
    """ repeating(var):
        takes in an individual word and checks whether or not it repeats multiple times.

        Return(s): True/False [Bool]
    """
    if var == len(var) * var[0]:
        return True
    else:
        return False


def tokenize(method, path, out_path, tokens, to_write):
    """ tokenize(method, path, out_path, tokens, to_write):
        tokenize takes in a method (personal or NLTK), path (file source), out_path (write source),
        and tokens (dict). tokenize will create tokens and add them into the tokens dictionary if
        they do not exist in the dictionary, or will increment the value by one if it exists. it will
        then write to a file the tokenized form of the text to a .txt file.

        Stopwords are used to remove extraneous words we do not need while processing this

        Finally, it will return a dictionary of tokens (storing tokens from all the scraped files) to
        scrape(type) so that we can calculate the total frequency and store all the tokens.

        Returns:    tokens [Dict(str : int)]     
    """
    tokenized = []

    with open(path, "r", encoding="utf-8", errors='ignore') as file:
        soup = Soup(file, "lxml")
        body = soup.get_text().lower()                                     # All lower case
        body = body.translate(body.maketrans("", "", string.punctuation))  # Getting rid of punctuation
        body = body.translate(body.maketrans("", "", string.digits))       # Getting rid of numbers
        body = re.sub(" +", " ", body)                                     # Getting rid of whitespace > 1                                             # Creating an array of all the words in string.

        # My "tokenizer" 
        if method == "personal":
            body_token = body.split()

            # Iterating to remove repeats e.g., "eeeee", "z", "p", and other stopwords
            filtered = filter(lambda var: not stop(var), body_token)
            filtered = filter(lambda var: not repeating(var), filtered)

            for word in filtered:        
                tokenized.append(word)
                if word not in tokens:    # If unique, add to dict.
                    tokens[word] = 1
                else:                     # If not unique, then increment.                                  
                    tokens[word] += 1
            
            if to_write == True:
                write(out_path, tokenized)

        # NLTK tokenizer
        if method == "nltk":
            nltk_token = nltk.tokenize.word_tokenize(body)

            if to_write == True: 
                write(out_path, nltk_token)

            for token in nltk_token:
                if token not in tokens:    # If unique, add to dict.
                    tokens[token] = 1
                else:                      # If not unique, then increment.
                    tokens[token] += 1

        return tokens


def scrape(type, num):
    """ scrape(type): 
        scrape will scrape HTML files in order to tokenize each file and write them to
        /out/ and will record the running time for each file being processed.

        Returns:    None [None] / Array of Dicts [Dict] (TF-IDF)
    """
    if type == "all":
        # Removing the time output files, if they exist.
        if os.path.exists("runtime.txt"):
            os.remove("runtime.txt")
        if os.path.exists("runtime_nltk.txt"):
            os.remove("runtime_nltk.txt")

        # Scrape through all files.
        for i in range(0, 2):
            tokens = {}
            cpu_start = time.process_time()
            real_start = time.time()
            bar = progressbar.ProgressBar()

            # Designating file name based on iteration.
            print("\nStarting Personal Tokenizer") if i == 0 else print("\nStarting NLTK tokenizer.")

            # Iterating through all 502 html files.
            for files in bar(os.listdir("files/")):
                # Necessary variables when iterating. 
                path = "files/" + files

                if path == "files/.DS_Store":
                    continue

                if i == 0:
                    out_path = "out/" + files.strip(".html") + ".txt"
                    ind_start = time.process_time()
                    tokens = tokenize("personal", path, out_path, tokens, True)
                    ind_end = time.process_time()
                    ind_final = ind_end - ind_start

                    # time output file for each file
                    with open("runtime.txt", "a") as file:
                        file.write(str(ind_final)+"\n")

                if i == 1:
                    out_path = "out/" + files.strip(".html") + "_nltk.txt"
                    ind_start = time.process_time()
                    tokens = tokenize("nltk", path, out_path, tokens, True)
                    ind_end = time.process_time()
                    ind_final = ind_end - ind_start

                    # time output file for each file
                    with open("runtime_nltk.txt", "a") as file:
                        file.write(str(ind_final)+"\n")

            # Organizing by frequency and tokens
            freq = {k: v for k, v in sorted(tokens.items(), key=lambda value: value[1], reverse=True)}  
            all_tokens = sorted(freq.keys())

            # Writing our final tokens and frequencies
            if i == 0:
                write("out/token.txt", all_tokens)
                write("out/freq.txt", freq)
            if i == 1:               
                write("out/token_nltk.txt", all_tokens)
                write("out/freq_nltk.txt", freq)

            # Calculating actual process time. 
            cpu_end = time.process_time()
            real_end = time.time()
            cpu_final = cpu_end - cpu_start 
            real_final = real_end - real_start
            print("Time elapsed (CPU): " + str(cpu_final) + "s")
            print("Time elapsed (Real): " + str(real_final) + "s")

    # Used for TFIDF
    elif type == "TF-IDF":
        # Initializing necessary variables.
        all_tokens = {}
        each = []
        remove = {}
        counter = 0

        # Iterate through all of the files in order.
        for files in sorted(os.listdir("files/")):
            tokens = {}
            path = "files/" + files
            tokens = tokenize("personal", path, None, tokens, False)
            all_tokens = Counter(all_tokens) + Counter(tokens)
            each.append(tokens)

            # If we have reached the max number of files to tokenize...
            if counter >= num:
                # Iterate through all the tokens and see if there is a token with only 1 counter.
                for tok, val in all_tokens.items():
                    if val <= 1:
                        remove[tok] = None
                
                # Iterate through array of all the dicts with tokenized documents
                for doc in each:
                    for tok in doc.copy().keys():
                        if tok in remove:
                            del doc[tok]

                return each

            else:
                counter+=1


def tf(arr):
    """ tf(): Calculating the Term Frequncy by doing n/sum(n)
        
        Return: tf_arr [arr(dict)]
    """
    
    # Initialize necessary variables.
    tf_arr = []

    # Iterate through the array...
    for doc in arr:
        tf_dict = {}
        count = len(doc)

        # Calculating term frequency: n/sum(n)
        for tok, val in doc.items():
            tf_dict[tok] = val / float(count)

        tf_arr.append(tf_dict) 
        del tf_dict

    return tf_arr


def idf(arr):
    """ idf(): Calculating the Inverse Document Frequency by doing log(n/df)
        
        Return: idf_arr [arr(dict)]
    """
    idf_arr = []
    len_corpus = len(arr) 

    # Iterate through corpus 
    for doc in arr:
        # Create a dictionary of tokens.
        idf_dict = dict.fromkeys(doc, 0)

        # Increment based on frequency of the item.
        for tok, val in doc.items():
            idf_dict[tok] += 1

        # Create the IDF of tokens in each document.
        for word, val in idf_dict.items():
            idf_dict[word] = math.log(len_corpus / float(val))

        idf_arr.append(idf_dict)
        del idf_dict

    return idf_arr


def tfidf():
    """ tfidf(): Calculating TFIDF by taking the dot product of TF and IDF

        Return:  None [None]
    """ 
    varying = [10, 20, 40, 80, 100, 200, 300, 400, 500]

    # Iterating through 10, 20, 40, ..., 300, 400, 500 times.
    for freq in varying:
        counter = 0
        time_start = time.time()
        print("\nBeginning TFIDF on " + str(freq) + " documents")

        arr = scrape("TF-IDF", freq)            # Tokenize the corpus.
        tf_arr = tf(arr)                        # Calculate TF
        idf_arr = idf(arr)                      # Calculate IDF

        # Calculate TF-IDF
        for doc in tf_arr:
            tfidf_dict = {}
            for tok, val in doc.items():
                tfidf_dict[tok] = val * idf_arr[counter][tok] 

            path = "out/" + str(counter) + ".txt"
            write(path, tfidf_dict)   
            counter += 1                        # Next document in IDF
        
        time_end = time.time()
        time_final = time_end - time_start
        print("Total time on " + str(freq) + " documents: " + str(time_final))


def main():
    """ main():
        bootstraps all the functions in this program.

        Returns: 0 (int) upon successful exit.
    """
    while True:
        # Validating user input
        while True:
            usr_input = int(input("\n1. Scrape all files\n2. Calculate Weight Terms.\n3. Exit\n\nProvide input: "))
            try: 
                if usr_input in range(1, 4):
                    break
            except ValueError:
                print("\nThat was an invalid input.")

        # Let's go to our destination!
        if usr_input == 1:
            scrape("all", None)
        elif usr_input == 2:
            tfidf()
        elif usr_input == 3:
            print("\nBye!")
            sys.exit(0)        

    return 0

main()
