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
import labeler_checker


topics = []
questions = []
test_questions = []
with open('labeler_sample.in', 'r') as f:
    first_line = f.readline().strip().split(" ")
    num_training, num_testing = int(first_line[0]), int(first_line[1])
    for index, line in enumerate(f):
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


parameters = {"C": [0.5, 1., 2., 3., 4., 5.],
              "class_weight": ['auto', 'balanced'],
              "k": [500, 750, 1000, 1500, 2000, 2500]
              }

scores = []
max_score = 0

for C in parameters["C"]:
    for class_weight in parameters["class_weight"]:
        for k in parameters["k"]:
            print "Starting test for parameters " + str(C) + ", " + str(k) + ", " + class_weight
            my_Pipeline = Pipeline([('tfidf', tfidf_transformer),
                                 ('select', SelectKBest(chi2, k=k)),
                                    ('clf',LogisticRegression(C=C, class_weight=class_weight)),
            ])
            my_OvR = OneVsRestClassifier(my_Pipeline, n_jobs=-1)
            fitted_Pipeline = my_OvR.fit(count_vect.fit_transform(questions), topics)
            probabilities_array = fitted_Pipeline.predict_proba(count_vect.transform(test_questions))


            with open("labeler_samples.myans", "w") as f:
                probabilities_array = fitted_Pipeline.predict_proba(count_vect.transform(test_questions))
                for probabilities in probabilities_array:
                    top = top_10_elements_helper(probabilities)
                    string_to_write = []
                    for i in reversed(top):
                        string_to_write.append(str(mlb.classes_[i]))
                    f.write(" ".join(string_to_write))
                    f.write("\n")

            with open("labeler_sample.ans") as f:
                with open("labeler_samples.myans") as g:
                    score = labeler_checker.check(f.read(), g.read())


            scores.append(str(C) + ", " + str(k) + ", " + class_weight + " leads to score" + str(score))
            max_score = max(max_score, score)
            print "Max score so far: " + str(max_score)

print scores