# proj.py - tokenization
#
# Author: Daniel Choo
# Date:   02/03/20


import os
import sys
import time
import json
import search


def main():
    """ main():
        bootstraps all the functions in this program.

        Menu:
        The menu provides the user an "interface" to interact with.
        They can:
            1. Scrape all the terms in all 504 HTML files.
            2. Conduct a TFIDF on all terms from 504 HTML files.
            3. Create an inverted index and postings file from aformentioned files.
            4. Exit the program.

        Command Line:
        In addition, the user can pass in parameters to the program. This completely changes the program
        from a scraper to an actual search. However, the user must first have a postings file of all the terms
        in the /out/ directory. The user can type ```python3 search.py term_x, term_y, term_z``` and retrieve the
        locations and weights for all the terms located at that specific location.

        Returns: 0 (int) upon successful exit [Menu], location of search terms (string) [CMD].
    """
    # Instantiating object.
    look = search.Search()

    # If we only executed the program with no args.
    if len(sys.argv) <= 1:
        while True:
            # Validating user input
            while True:
                usr_input = int(input("\n1. Scrape all files\n2. Calculate Weight Terms.\n3. Fixed File Lengths\n4. Exit\n\nProvide input: "))
                try: 
                    if usr_input in range(1, 5):
                        break
                except ValueError:
                    print("\nThat was an invalid input.")

            # Let's go to our destination!
            # Scraping all 504 HTML files (Phase 1)
            if usr_input == 1:
                look.scrape("all", None)

            # Calculating TFIDF (Phase 2)
            elif usr_input == 2:
                # Iterating through 10, 20, 40, ..., 300, 400, 500 times.
                varying = [10, 20, 40, 80, 100, 200, 300, 400, 500]

                for freq in varying:
                    look.tfidf(freq)

            # Creating an index (Phase 3)
            elif usr_input == 3:
                # Iterating through 10, 20, 40, ..., 300, 400, 500 times.
                varying = [10, 20, 40, 80, 100, 200, 300, 400, 500]

                for freq in varying:
                    look.indexing(freq)

            elif usr_input == 4:
                print("\nBye!")

    # Will go here if there are arguments (search queries.) (phase 4)
    else:
        index = {}

        # If our postings.json exists, its going to be blazing fast!
        if os.path.exists("out/postings.json"):
            # Instaniating variables.
            time_start = time.process_time()

            with open("out/postings.json", "r") as file:
                index = json.load(file)

            # Searching!
            look.lookup(index, sys.argv)

            # Print time. 
            time_end = time.process_time()
            time_final = time_end - time_start
            print("\nYour search took: " + str(time_final) + " seconds!")

        # If it does not exist, prompt the user whether or not they would like to proceed.
        else:
            x = "-1"
            try:
                while x != "y" and x != "n":
                    x = input("\nWARNING: Are you sure you would like to proceed? You are currently missing the postings.json file in your /out/ directory.\nMissing this file will cause your search runtime to be approximately 20 seconds.\nHowever, following this run, the next search will take approximately 0.33 seconds. (y/n): ")

                if x == "y":
                    # Instaniating variables.
                    time_start = time.process_time()
                    index = look.indexing(500)

                    # Searching!
                    look.lookup(index, sys.argv)

                    # Print time.
                    time_end = time.process_time()
                    time_final = time_end - time_start
                    print("\nYour search took: " + str(time_final) + " seconds!")

                else:
                    print("Bye!")
                    sys.exit(0)
            
            except Exception:
                print("That was an incorrect input. Please put in ")
        
    sys.exit(0)

main()