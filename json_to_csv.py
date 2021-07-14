# -*- coding: utf-8 -*-
"""
Convert json data to csv, keeping only 'user' and 'text' columns

@author: elizabeth witten
"""

import pandas as pd
import os

json_path = '../Data/1year_filtered_sorted.json.gz'
csv_path = '../Data/1year_filtered_sorted.csv.gz'

print('Getting chunks')

reader = pd.read_json(json_path, compression='infer', lines=True, chunksize=100000)
i = 1;
for chunk in reader:
    print(f'saving chunk #{i}')
    chunk = chunk[['user','text']]
    chunk.user = pd.DataFrame(chunk.user.values.tolist())['id']
    chunk.to_csv(csv_path, index=False, mode='a', header=not os.path.exists(csv_path), compression='infer')
    i = i + 1