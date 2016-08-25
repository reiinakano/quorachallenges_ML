# Enter your code here. Read input from STDIN. Print output to STDOUT
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

def get_inputs(num):
    for i in xrange(num):
        yield raw_input()

topics = []
questions = []
test_questions = []

first_line = raw_input().strip().split(" ")
num_training, num_testing = int(first_line[0]), int(first_line[1])
for index, line in enumerate(get_inputs(num_training*2 + num_testing)):
    if index < num_training * 2:
        if index % 2 == 0:
            topics.append(map(int, line.strip().split(" ")))
        else:
            questions.append(line.strip())
    else:
        test_questions.append(line.strip())

mlb = MultiLabelBinarizer()
topics = mlb.fit_transform(topics)
count_vect = CountVectorizer()

def top_10_elements_helper(array):
    ind = np.argpartition(array, -10)[-10:]
    return ind[np.argsort(array[ind])]

my_OvR = OneVsRestClassifier(SGDClassifier(loss="log", penalty='l2', alpha=1e-3, n_iter=5, random_state=42), n_jobs=-1)
my_Pipeline = Pipeline([('vect', count_vect),
                     #('tfidf', tfidf_transformer),
                     ('clf', my_OvR),
])
fitted_Pipeline = my_Pipeline.fit(questions, topics)
probabilities_array = fitted_Pipeline.predict_proba(test_questions)
for probabilities in probabilities_array:
    top = top_10_elements_helper(probabilities)
    string_to_write = []
    for i in reversed(top):
        string_to_write.append(str(mlb.classes_[i]))
    print " ".join(string_to_write)