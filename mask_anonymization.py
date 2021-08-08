# -*- coding: utf-8 -*-
"""
Anonymize tweets by replacing words to maximize the anonymization 

@author: elizabeth
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from transformers import pipeline
from joblib import load
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

in_test_csv_path = '../Data/1year_filtered_sorted_test.csv.gz'
out_test_csv_path = '../Data/1year_filtered_sorted_anon.csv.gz'
load_classfiers_path = '../joblib/clfs.joblib'
load_vectorizer_path = '../joblib/vectorizer.joblib'
load_transformer_path = '../joblib/tfidf.joblib'
mask_str = '[MASK]'
url_str = '[URL]'
mention_str = '@[USER]'
fill_mask_model = 'bert-base-uncased'


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

def anonymize_tweets(clfs: dict, vectorizer: CountVectorizer, transformer: TfidfTransformer):

    def get_anon_tweet(tweet: str, user: int):

        def replace_tweet_text(tweet: str, i_mask: int) -> str:            
            tweet_words = tweet.split()
            if tweet_words[i_mask].startswith('http://') or tweet_words[i_mask].startswith('https://'):
                tweet_words[i_mask] = url_str
            elif tweet_words[i_mask].startswith('@'):
                tweet_words[i_mask] = mention_str
            else:
                tweet_words[i_mask] = mask_str
                tweet_words[i_mask] = unmasker(' '.join(tweet_words))[0]['token_str']
            return ' '.join(tweet_words)

        def should_replace(orig_tweet: str, masked_tweet: str, i_word: int) -> bool:
            masked_tweet_words = masked_tweet.split()
            if masked_tweet_words[i_word] == url_str or masked_tweet_words[i_word] == mention_str:
                return True            
            orig_scores = []
            masked_scores = []
            X_orig_tf = get_text_features(vectorizer, transformer, [orig_tweet])
            X_masked_tf = get_text_features(vectorizer, transformer, [masked_tweet])
            for uid, clf in clfs.items():
                correct_class = 1 if uid == user else 0
                orig_scores.append(clf.predict_proba(X_orig_tf)[:,correct_class])
                masked_scores.append(clf.predict_proba(X_masked_tf)[:,correct_class])
            avg_orig_score = sum(orig_scores) / len(orig_scores)
            avg_masked_score = sum(masked_scores) / len(masked_scores)
            return avg_masked_score <= avg_orig_score

        tweet_word_count = len(tweet.split())
        for i_word in range(tweet_word_count):
            tweet_replaced = replace_tweet_text(tweet, i_word)
            if should_replace(tweet, tweet_replaced, i_word):
                tweet = tweet_replaced
        
        with lock:
            anon_tweets.append(tweet)
            anon_users.append(user)
        
    lock = Lock()  
    start = time.perf_counter()
    unmasker = pipeline('fill-mask', model=fill_mask_model, top_k=1, device=7)
    X_text, y_user = get_data(in_test_csv_path)
    num_tweets = len(X_text)

    anon_tweets = list()
    anon_users = list()
    with ThreadPoolExecutor(max_workers=10) as executor:            
        for i_tweet in range(num_tweets):
            executor.submit(get_anon_tweet, X_text[i_tweet], y_user[i_tweet])
    anon_df = pd.DataFrame(data={'text': anon_tweets, 'user': anon_users})
    pd.to_csv(anon_df, index=False)

    end = time.perf_counter()
    print(end-start)

def main():
    
    print('*** LOADING ***')
    clfs: dict = load(load_classfiers_path)
    vectorizer: CountVectorizer = load(load_vectorizer_path)
    transformer: TfidfTransformer = load(load_transformer_path)
    
    print('*** ANONYMIZING ***')
    anonymize_tweets(clfs, vectorizer, transformer)
    print('Done!')

main()
