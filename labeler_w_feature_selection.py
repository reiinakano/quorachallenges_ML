# Enter your code here. Read input from STDIN. Print output to STDOUT
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, chi2

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
tfidf_transformer = TfidfTransformer()

def top_10_elements_helper(array):
    ind = np.argpartition(array, -10)[-10:]
    return ind[np.argsort(array[ind])]

my_Pipeline = Pipeline([('tfidf', tfidf_transformer),
                     ('select', SelectKBest(chi2, k=1500)),
                        ('clf',LogisticRegression(C=4, class_weight='auto')),
])
my_OvR = OneVsRestClassifier(my_Pipeline, n_jobs=-1)
fitted_Pipeline = my_OvR.fit(count_vect.fit_transform(questions), topics)
probabilities_array = fitted_Pipeline.predict_proba(count_vect.transform(test_questions))
for probabilities in probabilities_array:
    top = top_10_elements_helper(probabilities)
    string_to_write = []
    for i in reversed(top):
        string_to_write.append(str(mlb.classes_[i]))
    print " ".join(string_to_write)