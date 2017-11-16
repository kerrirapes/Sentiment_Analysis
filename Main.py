# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:40:06 2017

@author: KRapes
MAIN
"""

import exploring
import Supervised_Learning
import pandas as pd
import os


print("Start")
percent_saved = 0.7
try:
    os.remove('df.pkl')
    os.remove('vocabulary.pkl')
except:
    pass
df = exploring.prepare_df_labeled(percent_saved)
df_human = df[df.cluster != -1]    
clf, score = Supervised_Learning.best_classifier(df_human, percent_saved)
clf, pruning_percent = Supervised_Learning.best_pruning_percent(clf, df_human)
df = exploring.prepare_df_labeled(pruning_percent)
df_machine = df[df.cluster == -1] 
df_machine = Supervised_Learning.predict_cluster(clf, df)
exploring.save_obj(df_machine, 'df_Machine_Labeled')
print("Done")