# -*- coding: utf-8 -*-
"""
Get ROC curves

@author: elizabeth
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, auc
from joblib import load

in_test_csv_path = '../Data/1year_filtered_sorted_test.csv.gz'
out_roc_path = '../Data/roc_curves.png'
load_classfiers_path = '../joblib/clfs.joblib'
load_vectorizer_path = '../joblib/vectorizer.joblib'
load_transformer_path = '../joblib/tfidf.joblib'
roc_lw = 2

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

def get_roc_curves(clfs, X_text_tf, y_user, all_userids, out_fig):

    def calc_user_curve(uid, clf):
        y_users_bin = np.asarray([u == uid for u in y_user], dtype=int)
        y_score = clf.predict_proba(X_text_tf)
        y_total_users_bin[uid] = y_users_bin
        y_total_scores[uid] = y_score[:,1]
        fpr[uid], tpr[uid], _ = roc_curve(y_users_bin, y_score[:,1])
        roc_auc[uid] = auc(fpr[uid], tpr[uid])

    def calc_micro_avg_curve():
        users = np.array(list(y_total_users_bin.values()))
        scores = np.array(list(y_total_scores.values()))
        fpr["micro"], tpr["micro"], _ = roc_curve(users.ravel(), scores.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    def calc_macro_avg_curve():
        print('\t\t\tAggregating all false positive rates...')
        all_fpr = np.unique(np.concatenate([fpr[u] for u in all_userids]))
        print('\t\t\tInterpolating all ROC curves...')
        mean_tpr = np.zeros_like(all_fpr)
        for u in all_userids:
            mean_tpr += np.interp(all_fpr, fpr[u], tpr[u])
        mean_tpr /= len(all_userids)
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    def export_roc_curves():
        print('Exporting graph...', flush=True)
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                label='micro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["micro"]),
                color='deeppink')
        plt.plot(fpr["macro"], tpr["macro"],
                label='macro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["macro"]),
                color='navy')
        plt.plot([0, 1], [0, 1], 'k--', lw=roc_lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curves for user classifiers')
        plt.legend(loc="lower right")
        plt.savefig(out_fig)

    print('Getting ROC curves', flush=True)
    start = time.perf_counter()    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_total_users_bin = dict()
    y_total_scores = dict()  

    print('\tCalculating ROC curves for each user classifier', flush=True)
    i_clf = 1
    for uid, clf in clfs.items():
        print(f'\r\t\t{i_clf}', end='', flush=True)
        calc_user_curve(uid, clf)
        i_clf = i_clf + 1
    
    print('\n\t\tMicro-average', flush=True)    
    calc_micro_avg_curve()

    print('\t\tMacro-average', flush=True)
    calc_macro_avg_curve()

    export_roc_curves()
    
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
    get_roc_curves(clfs, X_text_tf, y_user, clfs.keys(), out_roc_path)

    print('Done!')

main()