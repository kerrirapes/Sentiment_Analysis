# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:45:17 2017

@author: KRapes

Map labeled tweets to time
"""

import json
import pandas as pd
import Supervised_Learning
import label_building

json_location = "D:\Intelligens\challenge_en.json"


def load_json():
    with open(json_location, 'r') as json_data:
        json_lines = []
        for i,line in enumerate(json_data):
            #if i >= 3000:
               #break
            json_lines.append(json.loads(line))
       
    return pd.DataFrame.from_dict(json_lines)

Supervised_Learning.label_dataset()
df_labeled = pd.read_pickle('Group_1&2.pkl')
df = load_json()
df = label_building.relate_dfs(df, [df_labeled], column_name='prediction')

print(df.count())
columns = ['created_at', 'source',  'favorite_count', 'prediction']
df = df[columns]
print(df.head())

for col in columns:
    print(df.groupby(col).count())
