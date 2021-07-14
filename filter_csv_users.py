# -*- coding: utf-8 -*-
"""
Filter out all users who tweet fewer than 10 times

@author: elizabeth
"""

import pandas as pd
import os

in_csv_path = '../Data/1year_filtered_sorted.csv.gz'
out_csv_path = '../Data/1year_filtered_sorted_min10.csv.gz'
out_csv_counts_path = '../Data/1year_filtered_sorted_min10_counts.csv'
min_tweet_count = 10

total_counts = pd.DataFrame(columns=['size'])

print('Getting chunks for counts')
reader = pd.read_csv(in_csv_path, compression='infer', chunksize=1000000)
i = 1
for chunk in reader:
    print(f'getting counts from chunk #{i}')
    chunk_counts = chunk.groupby('user').size().to_frame('size')
    total_counts = total_counts.add(chunk_counts, fill_value=0)
    i = i + 1

print('getting list of users')
filtered_counts = total_counts.loc[total_counts['size'] >= min_tweet_count]
filtered_counts.to_csv(out_csv_counts_path, index=True)
include_users = filtered_counts.index.array

print('Getting chunks to filter')
reader = pd.read_csv(in_csv_path, compression='infer', chunksize=1000000)
i = 1
for chunk in reader:
    print(f'writing filtered chunk #{i}')
    chunk = chunk[chunk['user'].isin(include_users)]
    chunk.to_csv(out_csv_path, index=False, mode='a', header=not os.path.exists(out_csv_path), compression='infer')
    i = i + 1
