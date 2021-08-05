# -*- coding: utf-8 -*-
"""
Get Precision-Recall curves

@author: elizabeth
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from joblib import load

in_test_csv_path = '../Data/1year_filtered_sorted_test.csv.gz'
out_pr_path = '../Data/precision_recall.png'
load_classfiers_path = '../joblib/clfs.joblib'
load_vectorizer_path = '../joblib/vectorizer.joblib'
load_transformer_path = '../joblib/tfidf.joblib'
pr_lw = 2

def get_data(csv_path):
    print('Reading csv data... ', end = '', flush=True)
    start = time.perf_counter()
    df = pd.read_csv(csv_path, compression='infer', dtype={'user': int})
    X_text = df.text
    y_user = df.user
    end = time.perf_counter()
    print(f'{end-start:0.4f} seconds', flush=True)
    return X_text, y_user

def get_text_features(vectorizer, transformer, X_text):
    X_text_counts = vectorizer.transform(X_text)
    X_text_tf = transformer.transform(X_text_counts)
    return X_text_tf

def get_pr_curves(clfs, X_text_tf, y_user, all_userids, out_fig):

    def calc_user_curve(uid, clf):
        y_users_bin = np.asarray([u == uid for u in y_user], dtype=int)
        y_score = clf.predict_proba(X_text_tf)[:,1]
        y_total_users_bin.extend(y_users_bin)
        y_total_scores.extend(y_score)
        # precision[uid], recall[uid], _ = precision_recall_curve(y_users_bin, y_score)
        # avg_precision[uid] = average_precision_score(y_users_bin, y_score)

    def calc_micro_avg_curve():    
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_total_users_bin, y_total_scores)
        avg_precision["micro"] = average_precision_score(y_total_users_bin, y_total_scores, average="micro")

    def export_pr_curves():
        print('Exporting graph...', flush=True)
        plt.figure()     
        plt.step(recall['micro'], precision['micro'], 
                  where='post', 
                  label='micro-average precision scores (avg precision = {0:0.2f})'.format(avg_precision["micro"]))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Average precision score, micro-averaged over all users')
        plt.legend(loc="best")
        plt.savefig(out_fig)

    print('Getting Precision-Recall curves', flush=True)
    start = time.perf_counter()    
    precision = dict()
    recall = dict()
    avg_precision = dict()
    y_total_users_bin = list()
    y_total_scores = list()  

    print('\tCalculating PR curves for each user classifier', flush=True)
    i_clf = 1
    for uid, clf in clfs.items():
        print(f'\r\t\t{i_clf}', end='', flush=True)
        calc_user_curve(uid, clf)
        i_clf = i_clf + 1    
    
    print('\tMicro-average', flush=True)
    calc_micro_avg_curve()

    export_pr_curves()
    
    end = time.perf_counter()
    print(f'\tDone in {end-start:0.4f} seconds', flush=True)

def main():
    
    print('*** LOADING ***')
    clfs: dict = load(load_classfiers_path)
    vectorizer: CountVectorizer = load(load_vectorizer_path)
    transformer: TfidfTransformer = load(load_transformer_path)
    
    print('*** TESTING ***')
    X_text, y_user = get_data(in_test_csv_path)
    X_text_tf = get_text_features(vectorizer, transformer, X_text)
    get_pr_curves(clfs, X_text_tf, y_user, clfs.keys(), out_pr_path)

    print('Done!')

main()