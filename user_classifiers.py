# -*- coding: utf-8 -*-
"""
Train classifiers to predict if tweet is by specified user id

@author: elizabeth
"""

import time
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

print('Starting...')
in_csv_path = '../Data/1year_filtered_sorted_dev.csv.gz'
in_test_csv_path = '../Data/1year_filtered_sorted_test.csv.gz'
min_ngram_size = 1
max_ngram_size = 3
max_features = 4000

pd.set_option('display.max_colwidth', None)

print('Reading file...')
df = pd.read_csv(in_csv_path, compression='infer', dtype={'user': int})
y_user = df.user
X_text = df.text

print('Creating vectorizer...')
vectorizer = CountVectorizer(ngram_range=(min_ngram_size, max_ngram_size), max_features=max_features)
X_text_counts = vectorizer.fit_transform(X_text)

print('Creating tf-idf transformer...')
transformer = TfidfTransformer()
X_text_tf = transformer.fit_transform(X_text_counts)


print('Training classifier')
start = time.perf_counter()
clf = LinearSVC(verbose=1)
clf.fit(X_text_tf, y_user)
end = time.perf_counter()
print(f'\nTraining done in {end-start:0.4f} seconds')

print('Predicting')
start = time.perf_counter()
test_df = pd.read_csv(in_test_csv_path, compression='infer', dtype={'user':int})[:3]
test_text = test_df.text
test_counts = vectorizer.transform(test_text)
test_text_tf = transformer.transform(test_counts)
predicted = clf.predict(test_text_tf)
end = time.perf_counter()
print(f'\Predicting done in {end-start:0.4f} seconds')
print(predicted)
