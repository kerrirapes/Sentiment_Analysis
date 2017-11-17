# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:40:06 2017

@author: KRapes
MAIN
"""

import json_management
import Supervised_Learning
import os
from sklearn.model_selection import train_test_split
import graph


print("Preparing the Dataset")
pruning_percent = 0.7
try:
    os.remove('df.pkl')
    os.remove('vocabulary.pkl')
except:
    pass
df = json_management.prepare_df_labeled(pruning_percent)
print("The Dataset Contains {} Unique Messages".format(len(df)))
print("")
df_human = df[df.cluster != -1]
df_human, df_validation = train_test_split(df_human, test_size=0.10)
validation_idx = list(df_validation.index)

print("Finding the Best Classifier")   
clf, score = Supervised_Learning.best_classifier(df_human, pruning_percent)
print("")
print("The Best Classifier is:")
print(clf)
print("")
'''
print("Optimizing the Vocabulary")
clf, pruning_percent = Supervised_Learning.best_pruning_percent(clf, validation_idx)
print("")
'''
print("Labeling the Data")
df = json_management.prepare_df_labeled(pruning_percent)
df_machine = df[df.cluster == -1].copy()
df_machine = Supervised_Learning.predict_cluster(clf, df_machine)
print("{} Entries Have Now Been Labeled".format((len(df_machine))))
print("")

json_management.save_obj(df_machine, 'df_Machine_Labeled')

df_validation = Supervised_Learning.relate_dfs(df, df_validation)
df_validation = df_validation[df.cluster != -1]
score = clf.score(list(df_validation.features), list(df_validation.cluster))
print("The Final Validation Score is {}".format(round(score, 2)))

graph.cluster_results(df)
graph.cluster_results(df_human)
graph.cluster_results(df_machine)

print("Done")