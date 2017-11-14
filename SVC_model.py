# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:13:25 2017

@author: KRapes
"""

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
import exploring



df, features_master = exploring.preprocess_data()
df["cluster"] = -1.0

df_labeled = pd.read_pickle('labeled.pkl')

for i, row in df_labeled.iterrows():
    if row.text in list(df.text):
        idx = df.index[df['text'] == row.text]
        df.set_value(idx,'cluster', row.cluster)
        
df_train = df[df.cluster != -1]
df = df[df.cluster == -1]

X_train, X_test, y_train, y_test = train_test_split(list(df_train.features),
                                                    list(df_train.cluster),
                                                    test_size=0.33)



clf = SVC(kernel="linear", C=0.025)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("Score for the SVC model was {}".format(round(score,2)))

predictions = clf.predict(list(df.features))
df['prediction'] = predictions

print(df.groupby('prediction').count())

try:
    cgroups = range(2)
    for cgroup in cgroups:
        print("Messages from group {}".format(cgroup))
        for message in df.groupby('prediction').get_group(cgroup).text.head(10):
            print(message)
        print("")
except:
    print("Only one group")