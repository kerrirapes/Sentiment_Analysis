# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import json
from collections import Counter
from itertools import dropwhile
import numpy as np

json_location = "D:\Intelligens\challenge_en.json"


def load_json(json_location):
    with open(json_location, 'r') as json_data:
        json_lines = []
        for i,line in enumerate(json_data):
            if i > 100:
                break
            json_lines.append(json.loads(line))
       
    return pd.DataFrame.from_dict(json_lines)

def remove_nonalphanumeric(message):
    message = message.lower()
    delchar_table = {ord(c): None for c in message if c not in 'abcdefghijklmnopqrstuvwxyz0123456789 '}
    return message.translate(delchar_table)

def build_vocabulary(df):
    vocabulary = Counter()
    for message in df:
        message = remove_nonalphanumeric(message)
        vocabulary = vocabulary + Counter(message.split())
    
    for key, count in dropwhile(lambda key_count: key_count[1] >= (len(df) * .01), vocabulary.most_common()):
        del vocabulary[key]
    return vocabulary



df = load_json(json_location)
df = df[['text']]

vocabulary = build_vocabulary(df.text)
features_master = Counter(list(vocabulary.keys()))

df["features"] = [[0] * len(vocabulary)] * len(df)


for i,row in df.iterrows():
    message = remove_nonalphanumeric(row.text)
    features = Counter(message.split()) & features_master
    features = features + features_master
    features = np.array(list(features.values())) - 1
    df.set_value(i,'features',features)
    

print("")
print(df.head())
print(len(vocabulary))   

print("Done")

