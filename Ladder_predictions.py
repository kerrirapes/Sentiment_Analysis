# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:33:32 2017

@author: KRapes
"""
import os
import pandas as pd
import numpy as np
import exploring
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import  AdaBoostClassifier




df_labeled = pd.read_pickle('labeled.pkl')
df_labeled = df_labeled[df_labeled.cluster != -1]

df, features_master = exploring.preprocess_data()
df["cluster"] = -1.0
df["prediction"] = -1.0



for i, row in df_labeled.iterrows():
    if row.text in list(df.text):
        idx = df.index[df['text'] == row.text]
        df.set_value(idx[0],'cluster', row.cluster)
        
        
df_train = df[df.cluster != -1]


df = df[df.cluster == -1]

X_train, X_test, y_train, y_test = train_test_split(list(df_train.features),
                                                    list(df_train.cluster),
                                                    test_size=0.33)

names = ["SVC",  "Neural Net", "AdaBoost"]
    
classifiers = [SVC(gamma=2, C=1),
               MLPClassifier(alpha=1),
               AdaBoostClassifier()]

clfs = []

for name, clf in zip(names, classifiers):
    clfs.append(clf.fit(X_train, y_train))
    score = clf.score(X_test, y_test)
    print("Score for {} was {}".format(name, round(score,2)))

   
for i, row in df.iterrows():
    message = row.features
    predictions = []
    for name, clf in zip(names, clfs):
        predictions.append(clf.predict([message])[0])
    df.set_value(i,'prediction', sum(predictions))
    #print(predictions)

print(df.groupby('prediction').count())
try:
    cgroups = range(4)
    for cgroup in cgroups:
        print("Messages from group {}".format(cgroup))
        for message in df.groupby('prediction').get_group(cgroup).text.head(10):
            print(message)
        print("")
except:
    print("Only one group")
os.remove('df.pkl')
os.remove('vocabulary.pkl')      
        
        
        
        
        