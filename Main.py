# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:40:06 2017

@author: KRapes
MAIN
"""

import json_management
import Supervised_Learning
import pruning_dict
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans



print("Preparing the Dataset")
pruning_percent = 0.7
try:
    os.remove('df.pkl')
    os.remove('vocabulary.pkl')
except:
    pass
df = json_management.prepare_df_labeled(pruning_percent)
print("The Dataset Contains {} Unique Messages".format(len(df)))
df_human = df[df.cluster != -1]
df_human, df_validation = train_test_split(df_human, test_size=0.10)
print("There are {} training messages and {} validation messages".format(len(df_human),
                                                                             len(df_validation)))
print("")
validation_idx = list(df_validation.index)

print("Finding the Best Classifier")   
clf, score = Supervised_Learning.best_classifier(df_human, pruning_percent)
print("")
print("The Best Classifier is:")
print(clf)
print("")

print("Optimizing the Vocabulary")
clf, pruning_percent = Supervised_Learning.best_pruning_percent(clf, validation_idx)
print("")

print("Labeling the Data")
df = json_management.prepare_df_labeled(pruning_percent)
df_machine = df[df.cluster == -1].copy()
df_machine = Supervised_Learning.predict_cluster(clf, df_machine)
total = len(df_machine)
motivated = len(df[df['cluster'] == 0])
genuine = len(df[df['cluster'] == 1])
l_total = motivated + genuine
print("{} Entries Have Now Been Labeled and"
      " {} Entries Have Been Marked As Spam" .format(l_total,
                                                         total - l_total ))
print("Finacially Motivated: {}/{}  ({}%)       "
      "Genuine Expression: {}/{}  ({}%)".format(motivated,
                                                l_total,
                                             round(100 * motivated/l_total, 2),
                                             genuine,
                                             l_total,
                                             round(100 * genuine/l_total, 2)))
print("")

json_management.save_obj(df_machine, 'df_Machine_Labeled')

df_validation = Supervised_Learning.relate_dfs(df, df_validation)
df_validation = df_validation[df.cluster != -1]
score = clf.score(list(df_validation.features), list(df_validation.cluster))
print("The Final Validation Score is {}".format(round(score, 2)))


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

try:
    for cgroup in range(N):
        group = df_gen.groupby('class').get_group(cgroup)
        print("{} Total Messages from Group {}".format(len(group), cgroup))
        for message in df_gen.groupby('class').get_group(cgroup).text.head(8):
            print(message)
        print("")
except:
    print("Only one Group")


print("Done")