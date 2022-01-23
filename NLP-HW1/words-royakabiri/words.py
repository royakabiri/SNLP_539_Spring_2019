from typing import List, Text, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from heapq import nlargest
from collections import *
import re

def most_common(word_pos_path: Text,
                word_regex=".*",
                pos_regex=".*",
                n=10) -> List[Tuple[Text, int]]:
    """Finds the most common words and/or parts of speech in a file.

    :param word_pos_path: The path of a file containing part-of-speech tagged
    text. The file should be formatted as a sequence of tokens separated by
    whitespace. Each token should be a word and a part-of-speech tag, separated
    by a slash. For example: "The/at Hartsfield/np home/nr is/bez at/in 637/cd
    E./np Pelham/np Rd./nn-tl Aj/nn ./."

    :param word_regex: If None, do not include words in the output. If a regular
    expression string, all words included in the output must match the regular
    expression.

    :param pos_regex: If None, do not include part-of-speech tags in the output.
    If a regular expression string, all part-of-speech tags included in the
    output must match the regular expression.

    :param n: The number of most common words and/or parts of speech to return.

    :return: A list of (token, count) tuples for the most frequent words and/or
    parts-of-speech in the file. Note that, depending on word_regex and
    pos_regex (as described above), the returned tokens will contain either
    words, part-of-speech tags, or both.
    """
    file= open(word_pos_path,'r')
    textFile = file.read()
    file.close()
    #remove tabs and new lines
    textFile = textFile.strip()
    #split the text into 'word/pos' tokens by space
    wordPosList = textFile.split()
    
    posList=[]
    wordList=[]
    for wordPos in wordPosList:
        #split 'word/pos' tokens by slash
        splittedWordPos=wordPos.split("/")
        #append the last element to the pos list
        posList.append(splittedWordPos[-1])
        #append the rest to the word list
        wordList.append("/".join(splittedWordPos[:-1]))

    if word_regex==None and pos_regex:
        matchedPos=[]
        #loop through the pos list
        for word in posList:
            if re.match(pos_regex, word):
                matchedPos.append(word)
        #return the n most frequent pos tags
        mostCommonPos=Counter(matchedPos).most_common(n)
        return mostCommonPos

    elif pos_regex==None and word_regex:
        matchedWords=[]
        #loop through the word list
        for word in wordList:
            if re.match(word_regex, word):
                matchedWords.append(word)
        #return the n most frequent words
        mostCommonWords=Counter(matchedWords).most_common(n)
        return mostCommonWords

    elif word_regex and pos_regex:
        matchedWordPos=[]
        #loop through the wordPos list
        for wordPos in wordPosList:
            splitWordPos=wordPos.split("/")
            #join all the elements by slash except the last one (for cases where there are more than 1 slash)
            word="/".join(splitWordPos[:-1])
            pos=splitWordPos[-1]
            if re.match(word_regex, word) and re.match(pos_regex, pos):
                matchedWordPos.append(wordPos)
        #return the n most frequent words
        mostCommonWordPos=Counter(matchedWordPos).most_common(n)
        return mostCommonWordPos

    #when both the word-regex and non-regex are None
    return []


class WordVectors(object):
    wordVectorDict={}
    def __init__(self, word_vectors_path: Text):
        """Reads words and their vectors from a file.

        :param word_vectors_path: The path of a file containing word vectors.
        Each line should be formatted as a single word, followed by a
        space-separated list of floating point numbers. For example:

            the 0.063380 -0.146809 0.110004 -0.012050 -0.045637 -0.022240
        """
        self.word_vectors_path=word_vectors_path
        inFile = open (word_vectors_path, 'r')
        lines = inFile.readlines()
        inFile.close()
        for line in lines:
            line = line.strip()
            splittedLine=line.split()
            word=splittedLine[0]
            #join the vectors by space to make them str again
            wordVector=' '.join(splittedLine[1:])
            #change the vectors from str to array
            #specify sep argument as the single space was used to join the vectors
            wordVectorArray=np.fromstring(wordVector,sep=' ')
            #assign the word and its vector as key and value
            self.wordVectorDict[word]=wordVectorArray

    def average_vector(self, words: List[Text]) -> np.ndarray:
        """Calculates the element-wise average of the vectors for the given
        words.

        For example, if the words correspond to the vectors [1, 2, 3] and
        [3, 4, 5], then the element-wise average should be [2, 3, 4].

        :param words: The words whose vectors should be looked up and averaged.
        :return: The element-wise average of the word vectors.
        """
        vecList=[]
        for word in words:
            #append the vectors of the input words to list
            vecList.append(self.wordVectorDict[word])
        #average the vectors along the specified axis
        vectorAverage=np.average(vecList, axis=0)
        return vectorAverage

    def most_similar(self, word: Text, n=10) -> List[Tuple[Text, int]]:
        """Finds the most similar words to a query word. Similarity is measured
        by cosine similarity (https://en.wikipedia.org/wiki/Cosine_similarity)
        over the word vectors.

        :param word: The query word.words-royakabiri
        :param n: The number of most similar words to return.
        :return: The n most similar words to the query word.
        """
        similarityDict={}
        for key in self.wordVectorDict:
            if key!=word:
                #rehshape the vectors from 1D to 2D to get cosine-similarity
                similarityDict[key]=cosine_similarity(self.wordVectorDict[word].reshape(1, -1),self.wordVectorDict[key].reshape(1, -1))[0][0]
        #get the n most similar words to query (given their vectors)
        mostSimilarWords=nlargest(n,similarityDict.items(), key=lambda i:i[1])
        return mostSimilarWords
