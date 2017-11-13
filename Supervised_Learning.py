# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 23:35:08 2017

@author: KRapes

Supervised learning
"""

import pandas as pd
import numpy as np
import exploring
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


df, features_master = exploring.preprocess_data()
df["cluster"] = -1.0

df_labeled = pd.read_pickle('labeled.pkl')

for i, row in df_labeled.iterrows():
    if row.text in list(df.text):
        idx = df.index[df['text'] == row.text]
        df.set_value(idx,'cluster', row.cluster)
df = df[df.cluster != -1]

X_train, X_test, y_train, y_test = train_test_split(list(df.features), list(df.cluster), test_size=0.33, random_state=42)

print(len(df))
print(len(X_train), len(X_test), len(y_train), len(y_test))

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

