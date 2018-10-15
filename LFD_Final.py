from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import itertools
import numpy as np
from collections import Counter
from matplotlib.colors import Normalize
import time
import urllib.request
from yellowbrick.text import TSNEVisualizer
from mlxtend.plotting import plot_decision_regions
from sklearn.manifold import TSNE
from mlxtend.plotting import plot_decision_regions
from nltk import pos_tag
import string
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sys
from sklearn import svm

def read_corpus(corpus_file, test_file):
    reviews = []
    labels = []
    reviewsTest = []
    labelsTest = []
    lemmatizer = WordNetLemmatizer()
    # open the file of the user passed parameter directory with utf-8 encoding as value f
    with open(corpus_file, encoding='utf-8') as f:
        # Read each line
        print("Carrying out pre-processing on training data")
        for line in f:
            tokens = word_tokenize(line)
            tokens = [w.lower() for w in tokens]
            table = str.maketrans('', '', string.punctuation)  # Remove punctuation from each word
            stripped = [w.translate(table) for w in tokens]
            words = [word for word in stripped if word.isalpha()]  # Remove tokens that are not alphabetic
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if w not in stop_words]
            lemmatized = [lemmatizer.lemmatize(stem) for stem in words]  # Lemmatize
            reviews.append(lemmatized[3:])
            labels.append(tokens[1])

    with open(test_file, encoding='utf-8') as t:
        print ("Carrying out pre-processing on test data")
        for line in t:
            tokens = word_tokenize(line)
            tokens = [w.lower() for w in tokens]

            table = str.maketrans('', '', string.punctuation)  # Remove punctuation from each word
            stripped = [w.translate(table) for w in tokens]

            words = [word for word in stripped if word.isalpha()]  # Remove tokens that are not alphabetic

            stop_words = set(stopwords.words('english'))  # Remove stop words
            words = [w for w in words if w not in stop_words]
            lemmatized = [lemmatizer.lemmatize(stem) for stem in words]  # Lemmatize

            reviewsTest.append(lemmatized[3:])
            labelsTest.append(tokens[1])
    return reviews, labels, reviewsTest, labelsTest