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
from joblib import dump

in_train_csv_path = '../Data/1year_filtered_sorted_train.csv.gz'
dump_classfiers_path = '../joblib/clfs.joblib'
dump_vectorizer_path = '../joblib/vectorizer.joblib'
dump_transformer_path = '../joblib/tfidf.joblib'
minibatch_size = 100000
min_ngram_size = 1
max_ngram_size = 3
max_features = 4000
total_start = time.perf_counter()

def get_data(csv_path):
    print('Reading csv data... ', end = '', flush=True)
    start = time.perf_counter()
    df = pd.read_csv(csv_path, compression='infer', dtype={'user': int})
    X_text = df.text
    y_user = df.user
    end = time.perf_counter()
    print(f'{end-start:0.4f} seconds', flush=True)
    return X_text, y_user

def get_vectorizer_transformer(X_text):
    
    def fit_vectorizer():
        print('Creating vectorizer... ', end = '')
        start = time.perf_counter()
        vectorizer = CountVectorizer(ngram_range=(min_ngram_size, max_ngram_size), max_features=max_features)
        X_text_counts = vectorizer.fit_transform(X_text)
        end = time.perf_counter()
        print(f'{end-start:0.4f} seconds')
        return vectorizer, X_text_counts

    def fit_transformer(X_text_counts):
        print('Creating tf-idf transformer... ', end = '', flush=True)
        start = time.perf_counter()
        transformer = TfidfTransformer()
        transformer.fit_transform(X_text_counts)
        end = time.perf_counter()
        print(f'{end-start:0.4f} seconds', flush=True)
        return transformer

    vectorizer, X_text_counts = fit_vectorizer()
    transformer = fit_transformer(X_text_counts)
    return vectorizer, transformer

def get_text_features(vectorizer, transformer, X_text):
    X_text_counts = vectorizer.transform(X_text)
    X_text_tf = transformer.transform(X_text_counts)
    return X_text_tf

def get_minibatches(csv_path):
    print('Getting chunks for minibatches... ', end = '', flush=True)
    start = time.perf_counter()
    chunks = pd.read_csv(csv_path, compression='infer', dtype={'user': int}, chunksize=minibatch_size)
    end = time.perf_counter()
    print(f'{end-start:0.4f} seconds', flush=True)
    return chunks

def init_classifiers(y_user):
    print('Creating classifier for each user', flush=True)
    all_userids = np.unique(y_user)
    all_classes = np.array([0,1])
    clfs = { uid: MultinomialNB(alpha=0.01) for uid in all_userids }
    return clfs, all_userids, all_classes

def train_classifiers(chunks, clfs, all_classes, vectorizer, transformer):

    def partial_fit_classifiers():
        X_text_tf = get_text_features(vectorizer, transformer, chunk.text)
        y_user = chunk.user    
        start = time.perf_counter()
        i_clf = 1
        for uid, clf in clfs.items():
            print(f'\r\t\t{i_clf}', end='', flush=True)
            y_user_bin = np.asarray([u == uid for u in y_user], dtype=int)
            clf.partial_fit(X_text_tf, y_user_bin, classes=all_classes)
            i_clf = i_clf + 1    
        end = time.perf_counter()
        print(f'\t\t{end-start:0.4f} seconds', flush=True)

    print('Training classifiers with minibatches', flush=True)    
    i_batch = 1
    for chunk in chunks:
        print(f'\tminibatch {i_batch}... ', flush=True)
        partial_fit_classifiers()
        i_batch = i_batch + 1
    end = time.perf_counter()
    print(f'Total training time: {end-total_start:0.4f} seconds', flush=True)

def main():
    
    print('*** TRAINING ***')
    X_text, y_user = get_data(in_train_csv_path)
    vectorizer, transformer = get_vectorizer_transformer(X_text)
    chunks = get_minibatches(in_train_csv_path)
    clfs, all_userids, all_classes = init_classifiers(y_user)
    train_classifiers(chunks, clfs, all_classes, vectorizer, transformer)
    
    print('*** EXPORTING ***')
    dump(clfs, dump_classfiers_path)
    dump(vectorizer, dump_vectorizer_path)
    dump(transformer, dump_transformer_path)
    
    print('Done!')

main()