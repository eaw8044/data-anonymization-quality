# -*- coding: utf-8 -*-
"""
Train classifiers to predict if tweet is by specified user id (one classifier per id)
Use out-of-core learning 

@author: elizabeth
"""

import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

print('Starting...')
# pd.set_option('display.max_colwidth', None)
in_csv_path = '../Data/1year_filtered_sorted_train.csv.gz'
in_test_csv_path = '../Data/1year_filtered_sorted_test.csv.gz'
minibatch_size = 100000
min_ngram_size = 1
max_ngram_size = 3
max_features = 4000
total_start = time.perf_counter()

print('Reading data to fit vectorizer/transformer... ', end = '')
start = time.perf_counter()
full_df = pd.read_csv(in_csv_path, compression='infer', dtype={'user': int})
X_text = full_df.text
y_user = full_df.user
end = time.perf_counter()
print(f'{end-start:0.4f} seconds')

print('Creating vectorizer... ', end = '')
start = time.perf_counter()
vectorizer = CountVectorizer(ngram_range=(min_ngram_size, max_ngram_size), max_features=max_features)
X_text_counts = vectorizer.fit_transform(X_text)
end = time.perf_counter()
print(f'{end-start:0.4f} seconds')

print('Creating tf-idf transformer... ', end = '')
start = time.perf_counter()
transformer = TfidfTransformer()
X_text_tf = transformer.fit_transform(X_text_counts)
end = time.perf_counter()
print(f'{end-start:0.4f} seconds')

del full_df

print('Getting chunks for minibatches... ', end = '')
start = time.perf_counter()
chunks = pd.read_csv(in_csv_path, compression='infer', dtype={'user': int}, chunksize=minibatch_size)
end = time.perf_counter()
print(f'{end-start:0.4f} seconds')

print('Creating classifier for each user')
all_userids = np.unique(y_user)
all_classes = np.array([0,1])
clfs = { uid: MultinomialNB(alpha=0.01) for uid in all_userids }

print('Training classifiers with minibatches')    
i_batch = 1
for chunk in chunks:    
    print(f'\tminibatch {i_batch}... ')
    X_text_counts = vectorizer.transform(chunk.text)
    X_text_tf = transformer.transform(X_text_counts)
    y_user = chunk.user    
    start = time.perf_counter()
    i_clf = 1
    for uid, clf in clfs.items():
        print(f'\r\t\t{i_clf}', end='')
        y_user_bin = np.asarray([u == uid for u in y_user], dtype=int)
        clf.partial_fit(X_text_tf, y_user_bin, classes=all_classes)
        i_clf = i_clf + 1    
    end = time.perf_counter()
    print(f'\t\t{end-start:0.4f} seconds')
    i_batch = i_batch + 1
end = time.perf_counter()
print(f'Total training time: {end-total_start:0.4f} seconds')

# print('Predicting')
# start = time.perf_counter()
# test_df = pd.read_csv(in_test_csv_path, compression='infer', dtype={'user':int})[:10000]
# test_users = test_df.user
# test_text = test_df.text
# test_counts = vectorizer.transform(test_text)
# test_text_tf = transformer.transform(test_counts)
# pos_u = test_users[len(test_users)-1]
# clf = clfs[pos_u]
# test_users_bin = np.asarray([u == pos_u for u in test_users], dtype=int)
# accuracy = clf.score(test_text_tf, test_users_bin)
# end = time.perf_counter()
# print(f'Predicting done in {end-start:0.4f} seconds')
# print(f'Accuracy: {accuracy}')
