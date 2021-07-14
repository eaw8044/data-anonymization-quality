# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 10:13:26 2021

@author: elizabeth
"""

import pandas as pd
import os

in_csv_path = '../Data/1year_filtered_sorted_min10.csv.gz'
in_csv_counts_path = '../Data/1year_filtered_sorted_min10_counts.csv'

out_csv_test = '../Data/1year_filtered_sorted_test.csv.gz'   #25%
out_csv_dev = '../Data/1year_filtered_sorted_dev.csv.gz'     #25%
out_csv_train = '../Data/1year_filtered_sorted_train.csv.gz' #50%

print('reading counts')
count_df = pd.read_csv(in_csv_counts_path, index_col='user')
users = count_df.index.array

print('reading tweet data')
data_df = pd.read_csv(in_csv_path, compression='infer')

print('iterating over users')
for user in users:
    user_df = data_df.loc[data_df['user'] == user]
    portion = (int)(count_df.at[user, 'size'] // 4)
    user_df[:portion].to_csv(out_csv_test, index=False, mode='a', header=not os.path.exists(out_csv_test), compression='infer')
    user_df[portion : 2*portion].to_csv(out_csv_dev, index=False, mode='a', header=not os.path.exists(out_csv_dev), compression='infer')
    user_df[2*portion:].to_csv(out_csv_train, index=False, mode='a', header=not os.path.exists(out_csv_train), compression='infer')
