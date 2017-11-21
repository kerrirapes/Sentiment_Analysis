# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 22:05:05 2017

@author: KRapes
"""

import json_management
import Supervised_Learning
import pruning_dict
import os
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


df_machine = pd.read_pickle('df_Machine_Labeled.pkl')

df_gen = df_machine[df_machine.prediction == 1]
vocabulary = pruning_dict.build_vocabulary(df_gen.text)
#vocabulary = pruning_dict.prune_vocab(vocabulary, ['df_Machine_Labeled.pkl'] , pruning_percent)
features_master = Counter(list(vocabulary.keys()))
df_gen["features"] = [[0] * len(vocabulary)] * len(df_gen)
df_gen = json_management.label_features(df_gen, features_master)
df2 = json_management.create_feature_dataframe(df_gen, features_master)
N = json_management.cluster_search(df2)

print(N)

clusterer = KMeans(n_clusters=N)
clusterer.fit(df2)
preds = clusterer.predict(df2)
df_gen['class'] = preds

#try:
for cgroup in range(N):
    group = df_gen.groupby('class').get_group(cgroup)
    print("{} Total Messages from Group {}".format(len(group), cgroup))
    for message in df_gen.groupby('class').get_group(cgroup).text.head(8):
        print(message)
    print("")
#except:
print("Only one Group")


print("Done")