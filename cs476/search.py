# search.py
#
# Author: Daniel Choo
# Date:   04/17/20


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
from collections import Counter
from collections import OrderedDict
from matplotlib import pyplot as plt
from bs4 import BeautifulSoup as Soup
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering


class Search:
    """ Search:
        TBD.
    """

    def __init__(self):
        # Instantiating cached global variable
        self.stopset = stoplist.stopwords()
        self.validation = {"y": None, "Y": None, "n": None, "N": None, "": None}


    def write(self, path, contents, type):
        """ write(self, path, contents, type):
            takes in a path and contents, and writes/dumps the file as a JSON.

            Returns:    Nothing [None]
        """
        try:
            if type == "None":
                with open(path, "w+", encoding="utf-8") as file:
                    file.write(json.dumps(contents, ensure_ascii=False, indent=2))

            if type == "ASCII":
                with open(path, "w+") as file:
                    for lines in contents:
                        file.write(json.dumps(lines))
                        file.write("\n")

            if type == "pretty":
                with open(path, "w+", encoding="utf-8") as file:
                    file.write(json.dumps(contents, indent=None, separators=(",", ":")))

        except Exception:
            raise RuntimeError("There was something wrong with writing the file.")


    def stop(self, var):
        """ stop(var):
            takes in a word and checks whether or not it is in stopset.

            Return(s): True/False [Bool]
        """
        if var in self.stopset:
            return True
        else:
            return False


    def repeating(self, var):
        """ repeating(var):
            takes in an individual word and checks whether or not it repeats multiple times.

            Return(s): True/False [Bool]
        """
        if var == len(var) * var[0]:
            return True
        else:
            return False


    def tokenize(self, method, path, out_path, tokens, to_write):
        """ tokenize(self, method, path, out_path, tokens, to_write):
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
            body = re.sub(" +", " ", body)                                     # Getting rid of whitespace > 1

            # My "tokenizer" 
            if method == "personal":
                body_token = body.split()

                # Iterating to remove repeats e.g., "eeeee", "z", "p", and other stopwords
                filtered = filter(lambda var: not self.stop(var), body_token)
                filtered = filter(lambda var: not self.repeating(var), filtered)

                for word in filtered:        
                    tokenized.append(word)
                    if word not in tokens:    # If unique, add to dict.
                        tokens[word] = 1
                    else:                     # If not unique, then increment.
                        tokens[word] += 1

                if to_write is True:
                    self.write(out_path, tokenized, "None")

            # NLTK tokenizer
            if method == "nltk":
                nltk_token = nltk.tokenize.word_tokenize(body)

                if to_write is True:
                    self.write(out_path, nltk_token, "None")

                for token in nltk_token:
                    if token not in tokens:    # If unique, add to dict.
                        tokens[token] = 1
                    else:                      # If not unique, then increment.
                        tokens[token] += 1

            return tokens


    def scrape(self, type, num):
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
                        tokens = self.tokenize("personal", path, out_path, tokens, True)
                        ind_end = time.process_time()
                        ind_final = ind_end - ind_start

                        # time output file for each file
                        with open("runtime.txt", "a") as file:
                            file.write(str(ind_final)+"\n")

                    if i == 1:
                        out_path = "out/" + files.strip(".html") + "_nltk.txt"
                        ind_start = time.process_time()
                        tokens = self.tokenize("nltk", path, out_path, tokens, True)
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
                    self.write("out/token.txt", all_tokens, "None")
                    self.write("out/freq.txt", freq, "None")
                if i == 1:               
                    self.write("out/token_nltk.txt", all_tokens, "None")
                    self.write("out/freq_nltk.txt", freq, "None")

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

                # Would keep messing up my files by one.
                if files == ".DS_Store":
                    continue

                else:
                    tokens = self.tokenize("personal", path, None, tokens, False)
                    all_tokens = Counter(all_tokens) + Counter(tokens)
                    each.append(tokens)

                    # If we have reached the max number of files to tokenize...
                    if counter >= num:
                        # Iterate through all the tokens and see if there is a token with only 1 counter.
                        for tok, val in all_tokens.items():
                            if val == 1:
                                remove[tok] = None

                        # Iterate through array of all the dicts with tokenized documents
                        for doc in each:
                            for tok in doc.copy().keys():
                                if tok in remove:
                                    del doc[tok]
                        return each

                    else:
                        counter+=1


    def tf(self, arr):
        """ tf():
            Calculating the Term Frequncy by doing n/sum(n)

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


    def idf(self, arr):
        """ idf():
            Calculating the Inverse Document Frequency by doing log(n/df)

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


    def tfidf(self, freq):
        """ tfidf():
            Calculating TFIDF by taking the dot product of TF and IDF

            Return:  None [None]
        """
        # Initializing variables
        if freq == 500:
            tfidf_list = []

        counter = 0
        time_start = time.time()
        print("\nBeginning TFIDF on " + str(freq) + " documents")

        arr = self.scrape("TF-IDF", freq)            # Tokenize the corpus.
        tf_arr = self.tf(arr)                        # Calculate TF
        idf_arr = self.idf(arr)                      # Calculate IDF

        # Calculate TF-IDF
        for doc in tf_arr:
            tfidf_dict = {}
            for tok, val in doc.items():
                tfidf_dict[tok] = val * idf_arr[counter][tok]

            path = "out/" + str(counter) + ".txt"
            if freq == 500:
                tfidf_list.append(tfidf_dict)
            self.write(path, tfidf_dict, "None")
            counter += 1                            # Next document in IDF

        time_end = time.time()
        time_final = time_end - time_start
        print("Total time on " + str(freq) + " documents: " + str(time_final))
        return tfidf_list


    def indexing(self, freq):
        """ indexing(self):
            Creating an index by creating two files: 
            1. a dictionary that contains the word, # of docs that has this word,
               and the location where this is first found in the postings file.

            2. postings file that contains the document id and normalized weight 
               of the word in the document.

            Return:    None [None]
        """

        # Initializing variables.
        corpus = {}
        postings = {}
        doc_counter = 1
        line_counter = 1
        counter = 0
        time_start = time.time()
        print("\nConstructing Index on " + str(freq) + " documents")

        arr = self.scrape("TF-IDF", freq)            # Tokenize the corpus.
        tf_arr = self.tf(arr)                        # Calculate TF

        # Iterate through our term frequencies
        for doc in tf_arr:
            # Initializing variables.
            check = {}

            # Iterate through our documents
            for term, val in doc.items():
                # If the term is not in our corpus yet and we have seen it in current doc
                if term not in corpus and term not in check:
                    corpus[term] = [1, 0]
                    postings[term] = [(doc_counter, val)]
                    check[term] = None
                # If the term is in our corpus and we have not seen it in current doc
                elif term in corpus and term not in check:
                    corpus[term][0] += 1
                    postings[term].append((doc_counter, val))
                # If the term is in our corpus, and we have seen it already
                else:
                    continue

            # Keeping track of which document we're dealing with
            doc_counter += 1
            counter += 1

        # Save a json copy of our postings
        if freq == 500:
            self.write("out/postings.txt", postings, "None")

        # Location of the first record for that word in the postings file.
        # Since the postings file is in order, should work with a counter.
        for terms, val in postings.items():
            if terms in corpus.keys():
                corpus[terms][1] = line_counter
                line_counter += 1

        corpus = sorted(corpus.items())
        postings_list = [(k,v) for k,v in postings.items()]

        # Write
        out_postings = "out/postings_" + str(freq) + ".txt"
        out_corpus = "out/corpus_" + str(freq) + ".txt"
        self.write(out_postings, postings_list, "ASCII")
        self.write(out_corpus, corpus, "ASCII")

        time_end = time.time()
        time_final = time_end - time_start
        print("Total time indexing " + str(freq) + " documents: " + str(time_final))

        return postings


    def lookup(self, index, argv):
        """ lookup(self, index, argv):
            This function takes in the index that we have created and will check to see if a specific keyword exists in our
            index. It will also return the html files in weighted order instead of file order.

            Return: None [None]
        """
        # Iterate starting from second argument as first argument is our program.
        for args in sys.argv[1:]:
            print("\n" + args)
            print("-" * len(args))
            if args in index:
                counter = 0

                # Makes more sense for our weights to be in order.
                for tup in sorted(index[args], key=lambda x : x[1], reverse=True):
                    for i in tup:
                        # Pretty print.
                        if i < 10:
                            print("00" + str(i) + ".html " + str(tup[1]))
                        elif i < 100:
                            print("0" + str(i) + ".html " + str(tup[1]))
                        else:
                            print(str(i) + ".html " + str(tup[1]))
                        counter += 1
                        break

                    # Only print top 10 queries.
                    if counter >= 10:
                        break
            else:
                print("Sorry the word " + str(args) + " was not found in our index. :(")


    def open_postings(self):
        """ open_postings(self):
            Reads in the postings file in the out/ dir and makes an
            ordered dict from the file.

            Return(s):    lexicon [OrderedDict]
        """
        # Initializing variables.
        lexicon = OrderedDict()

        with open("out/postings.txt", "r") as file:
            # Formatting our tokens
            for line in file:
                line = line.translate(line.maketrans("", "", string.punctuation))  # Getting rid of punctuation
                line = line.translate(line.maketrans("", "", string.digits))       # Getting rid of numbers
                line = re.sub(" +", "", line)                                      # Getting rid of whitespace > 1
                line = line.strip("\n")                                            # Getting rid of escape sequence
                lexicon.update({line : None})                                      # Updating our OrderedDict

        return lexicon


    def cosine_sim(self, a, b):
        """ cosine_sim
            Calculates cosine similarity.

            Return(s):   cosine similarity [float (-1, 1)]
        """
        return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


    def matrix(self, lexicon):
        """ matrix(self, lexicon):
            Conducting our cosine similarity matrix construction/multiplications here.

            Return(s):    None [None]
        """
        # Initializing variables.
        matrix = []
        similarity = []
        bar = progressbar.ProgressBar()

        # Create tfidf array of dicts
        tfidf = self.tfidf(500)

        # Constructing the matrix of tfidf in respect to the words
        for docs in range(0, 500):
            doc_arr = []
            for word in lexicon.keys():
                # If the word exists in our sorted lexicon, append tfidf.
                if word in tfidf[docs].keys():
                    doc_arr.append(tfidf[docs][word])
                # Else append 0
                else:
                    doc_arr.append(0)
            # Append the tdidf of the document into here.
            matrix.append(doc_arr)

        # Construct similarity matrix
        print("\nConstructing similarity matrix...")
        for i in bar(range(0, 500)):
            # Initialize a list of zeroes.
            similarity.append([0]*500)

            for j in range(0, 500):
                # If it is the same document, then append a 1 because its exactly the same.
                if i == j:
                    similarity[i][j] = 1
                # Matrix is located in upper triangular in form. Thus, matrix is symmetric.
                elif i > j:
                    similarity[i][j] = similarity[j][i]
                # Else, do cosine similarity between the two documents.
                else:
                    similarity[i][j] = self.cosine_sim(matrix[i], matrix[j])

        self.write("out/similarity.txt", similarity, "None")
        self.clustering(similarity)


    def plot_dendrogram(self, model, **kwargs):
        """ plot_dendogram(self, model, **kwargs)
            Credit: https://github.com/scikit-learn/scikit-learn/blob/70cf4a676caa2d2dad2e3f6e4478d64bcb0506f7/examples/cluster/plot_hierarchical_clustering_dendrogram.py
            Wanted to find a way to properly plot out the clustering.

            Return(s):    None [None]
        """
        # Children of hierarchical clustering
        children = model.children_

        # Distances between each pair of children
        # Since we don't have this information, we can use a uniform one for plotting
        distance = np.arange(children.shape[0])

        # The number of observations contained in each cluster level
        no_of_observations = np.arange(2, children.shape[0]+2)

        # Create linkage matrix and then plot the dendrogram
        linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)


    def clustering(self, similarity):
        """ clustering(self):
            Perform clustering stuff here.

            Return(s):    None [None]
        """
        cpu_start = time.process_time()

        model = AgglomerativeClustering(affinity='cosine', n_clusters=None, linkage='average', compute_full_tree=True, distance_threshold=0.4).fit(similarity)
        print(len(model.labels_))
        print(model.labels_)
        plt.title('Hierarchical Clustering Dendrogram')
        self.plot_dendrogram(model, labels=model.labels_)
        plt.show()

        cpu_end = time.process_time()
        print("\nIt took " + str(cpu_end - cpu_start) + " seconds to cluster!")


    def cluster_strapper(self):
        """ cluster_strapper(self):
            Get ready to do some matrix multiplication and clustering!

            Return(s):  None [None]
        """
        # If we have already calculated the similarity matrix, cluster!
        if os.path.exists("out/similarity.txt"):
            with open("out/similarity.txt", "r") as file:
                similarity = json.load(file)
                self.clustering(similarity)

        # If we have the postings file, create sim matrix and cluster!
        elif os.path.exists("out/postings.txt"):
            print("Creating TFIDF Matrix (Warning: will take approximately 30 minutes")
            print("to create the similarity matrix.)")
            # Open postings file.
            lexicon = self.open_postings()

            # Create our similarity matrix.
            self.matrix(lexicon)

        # Else, create postings file, sim matrix, then cluster!
        else:
            try:
                usr_input = input("You are missing the postings file. Would you like to create the postings file? (y/n): ")

                if usr_input in self.validation.keys():
                    if usr_input == "y" or usr_input == "" or usr_input == "Y":
                        # Create the postings file
                        self.indexing(500)
                        lexicon = self.open_postings()
                        print("Creating the matrix will take approximately 30 minutes.")
                        self.matrix(lexicon)

                    elif usr_input == "n" or usr_input == "N":
                        print("\nBye!\n")
                        sys.exit(0)

            except Exception:
                print("That was not a valid input. Please try again.\n")