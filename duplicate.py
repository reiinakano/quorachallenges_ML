import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier

raw_question_data = []
training_pairs = []
training_labels = []
testing_data = []

N = int(raw_input().strip())
for i in xrange(N):
    raw_question_data.append(json.loads(raw_input()))
N = int(raw_input().strip())
for i in xrange(N):
    line = raw_input()
    training_pairs.append(tuple(line.split()[:2]))
    training_labels.append(int(line.split()[2]))
N = int(raw_input().strip())
for i in xrange(N):
    testing_data.append(tuple(raw_input().split()[:2]))

topics = {}
for question in raw_question_data:
    for topic in question["topics"]:
        topics[topic["name"]] = topic["followers"]
    if question["context_topic"]:
        topics[question["context_topic"]["name"]] = question["context_topic"]["followers"]

question_dictionary = {}
for question in raw_question_data:
    topic_set = set()
    if question["context_topic"]:
        topic_set.add(question["context_topic"]["name"])
    for topic in question["topics"]:
        topic_set.add(topic["name"])
    question_dictionary[question["question_key"]] = {"age": question["age"],
                                            "follow_count": question["follow_count"],
                                            "question_text": question["question_text"],
                                            "view_count": question["view_count"],
                                            "topic_set": topic_set
                                           }
raw_question_data = None

tfidf_vector = CountVectorizer()
tfidf_vector.fit([value["question_text"] for value in question_dictionary.itervalues()])

def convert(question_key_pairs, vector):
    # use vector to convert question text into bag of words representation
    question_features = [question_dictionary[key1]["question_text"] + " " + question_dictionary[key2]["question_text"] for key1, key2 in question_key_pairs]
    question_features = vector.transform(question_features)

    # add age, follow count, and view count of each question into the features
    additional_features = [[question_dictionary[key1]["age"], question_dictionary[key2]["age"],
                           question_dictionary[key1]["follow_count"], question_dictionary[key2]["follow_count"],
                           question_dictionary[key1]["view_count"], question_dictionary[key2]["view_count"]] for key1, key2 in question_key_pairs]
    additional_features = coo_matrix(additional_features)
    question_features = hstack([question_features, additional_features])

    # add "number of common topics" feature
    additional_features = []
    for key1, key2 in question_key_pairs:
        common = len(question_dictionary[key1]["topic_set"].intersection(question_dictionary[key2]["topic_set"]))
        additional_features.append([common])
    additional_features = coo_matrix(additional_features)
    question_features = hstack([question_features, additional_features])

    return question_features

training_data = convert(training_pairs, tfidf_vector)
to_predict = convert(testing_data, tfidf_vector)

clf = LogisticRegression(C=100)
clf = clf.fit(training_data, training_labels)
answers = clf.predict(to_predict)

for pair, label in zip(testing_data, answers):
    print pair[0] + " " + pair[1] + " " + str(label)