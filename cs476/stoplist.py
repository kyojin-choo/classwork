def stopwords():
    stopwords = set()
    with open("stoplist.txt", "r") as file:
        for word in file:
            stopwords.add(word.strip("\n"))
    return stopwords