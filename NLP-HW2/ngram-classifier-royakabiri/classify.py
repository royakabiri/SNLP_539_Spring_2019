from typing import Iterator, Iterable, Tuple, Text, Union
import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

NDArray = Union[np.ndarray, spmatrix]


def read_smsspam(smsspam_path: str) -> Iterator[Tuple[Text, Text]]:
    """Generates (label, text) tuples from the lines in an SMSSpam file.

    SMSSpam files contain one message per line. Each line is composed of a label
    (ham or spam), a tab character, and the text of the SMS. Here are some
    examples:

      spam	85233 FREE>Ringtone!Reply REAL
      ham	I can take you at like noon
      ham	Where is it. Is there any opening for mca.

    :param smsspam_path: The path of an SMSSpam file, formatted as above.
    :return: An iterator over (label, text) tuples.
    """
    #open the text file
    textFile=open(smsspam_path, 'r')
    #read the text file line by line
    lines=textFile.readlines()
    textFile.close()
    smsspamList=[]
    for line in lines:
        #remove new lines
        line=line.strip("\n")
        #split by tab to separate the label and the text
        splittedline=line.split("\t")
        #append the label and the text as a tuple to a list
        smsspamList.append(splittedline)
    #return the new list
    return  smsspamList

class TextToFeatures:
    def __init__(self, texts: Iterable[Text]):
        """Initializes an object for converting texts to features.

        During initialization, the provided training texts are analyzed to
        determine the vocabulary, i.e., all feature values that the converter
        will support. Each such feature value will be associated with a unique
        integer index that may later be accessed via the .index() method.

        It is up to the implementer exactly what features to produce from a
        text, but the features will always include some single words and some
        multi-word expressions (e.g., "need" and "to you").

        :param texts: The training texts.
        """
        #extract numerical features from text content
        #also tune the parameters to get higher accuracy and F1
        #binarizing the features to get higher accuracy and F1
        #character-based analyzer gives higher F1 than word-based
        #in char-based, ngram should have large range
        self.vectorizer = CountVectorizer(binary=True, ngram_range=(1, 6),analyzer='char') 
        #learn the vocabulary dictionary of all tokens
        self.vectorizer.fit(texts)

    def index(self, feature: Text):
        #mapping of terms to feature indices by vocabulary_ attribute of the vectorizer
        feature_index = self.vectorizer.vocabulary_.get(feature) 
        return feature_index  

    def __call__(self, texts: Iterable[Text]) -> NDArray:
        #Transform documents to term-document matrix 
        featureMatrix=self.vectorizer.transform(texts)
        return featureMatrix 

class TextToLabels:
    def __init__(self, labels: Iterable[Text]):
        """Initializes an object for converting texts to labels.

        During initialization, the provided training labels are analyzed to
        determine the vocabulary, i.e., all labels that the converter will
        support. Each such label will be associated with a unique integer index
        that may later be accessed via the .index() method.

        :param labels: The training labels.
        """
        #transform non-numerical/text labels to numerical labels
        self.encoder=preprocessing.LabelEncoder()
        #fit label encoder
        self.encoder.fit(labels)

    def index(self, label: Text) -> int:
        """Returns the index in the vocabulary of the given label.

        :param label: A label
        :return: The unique integer index associated with the label.
        """
        #by classes_, hold the label for each class, it is 'array' type
        #we have to convert it to a list, so we can index to it
        labelIndex =(list(self.encoder.classes_)).index(label)
        return labelIndex  

    def __call__(self, labels: Iterable[Text]) -> NDArray:
        """Creates a label vector from a sequence of labels.

        Each entry in the vector corresponds to one of the input labels. The
        value at index j is the unique integer associated with the jth label.

        :param labels: A sequence of labels.
        :return: A vector, with one entry for each label.
        """
        #convert labels to a label vector
        labelVector=self.encoder.transform(labels)
        return labelVector

class Classifier:
    def __init__(self):
        """Initalizes a logistic regression classifier.
        """
        #tune the parameters to get higher accuracy and F1
        #l1 regularization works better than l2 here
        #parameter C for regularization strength (smaller values specify stronger regularization)
        self.logisticRegr = LogisticRegression(penalty='l1',solver='liblinear', C=5)

    def train(self, features: NDArray, labels: NDArray) -> None:
        """Trains the classifier using the given training examples.

        :param features: A feature matrix, where each row represents a text.
        Such matrices will typically be generated via TextToFeatures.
        :param labels: A label vector, where each entry represents a label.
        Such vectors will typically be generated via TextToLabels.
        """
        #fit the model according to the given training data (both features and lables)
        self.logisticRegr.fit(features, labels)

    def predict(self, features: NDArray) -> NDArray:
        """Makes predictions for each of the given examples.

        :param features: A feature matrix, where each row represents a text.
        Such matrices will typically be generated via TextToFeatures.
        :return: A prediction vector, where each entry represents a label.
        """
        #predict class labels for input features
        predictionVector = self.logisticRegr.predict(features)
        return predictionVector



