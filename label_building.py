# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 20:55:25 2017

@author: KRapes
Polling Supervised Learnt Algorthims to build labels
"""

import pandas as pd
import exploring
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import  AdaBoostClassifier
from collections import Counter
import time
import os


def relate_dfs(df_features, df_clusters):
    for label in df_clusters:
        for i, row in label.iterrows():
            if row.text in list(df_features.text):
                idx = df_features.index[df_features['text'] == row.text]
                df_features.set_value(idx,'cluster', row.cluster)
    return df_features

def survey_clfs(message):
    predictions = []
    for name, clf in zip(names, clfs):
        predictions.append(clf.predict([message])[0])
    return predictions

def label_dataset(df):
    for i, row in df.iterrows():
        predictions = survey_clfs(row.features)
        keys = list(Counter(predictions))
        if len(keys) <= 1:
            df.set_value(i,'cluster', keys[0])
    return df

def group_predict(df):
    predictions = []
    for i, row in df.iterrows():
        predictions.append(Counter(survey_clfs(row.features)).most_common(1)[0][0])     
    return predictions

start = time.time()

df_labeled = pd.read_pickle('labeled.pkl')
df_labeled, df_validation = train_test_split(df_labeled, test_size=0.10)
df_labeled = df_labeled[df_labeled.cluster != -1]
df_validation = df_validation[df_validation.cluster != -1]

LM_final = 1
LM_previous = 0
count = 0 
while LM_final - LM_previous > 0:
    count += 1
    print("Count: {}".format(count))
    
    df_ml = pd.read_pickle('machine_labeled.pkl')
    if count <= 1:
        df_ml = pd.DataFrame(columns=['text', 'features', 'cluster'])
        df_ml.to_pickle('machine_labeled.pkl')
    LM_previous = len(df_ml[df_ml.cluster != -1])
    print("Machine Labeled Messages Size: {}".format(LM_previous))
    
    df, features_master = exploring.preprocess_data()
    df["cluster"] = -1.0
    
    df_validation = relate_dfs(df, [df_validation])
    df_validation = df_validation.copy()
    df_validation = df_validation[df_validation.cluster != -1]
    #print(df_validation.groupby('cluster').count())
    df = relate_dfs(df, [df_labeled, df_ml])

    df_train =  df[df.cluster != -1]       
    

    
    X_train, X_test, y_train, y_test = train_test_split(list(df_train.features),
                                                        list(df_train.cluster),
                                                        test_size=0.33,
                                                        random_state=42)
    
    names = ["Linear SVM",  "Neural Net", "AdaBoost"]
    
    classifiers = [
        SVC(gamma=2, C=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier()]
    
    clfs = []
    
    for name, clf in zip(names, classifiers):
            clfs.append(clf.fit(X_train, y_train))
            score = clf.score(X_test, y_test)
            print("Score for {} was {}".format(name, round(score,2)))
            
    df_validation['prediction'] = group_predict(df_validation)
    accuracy = df_validation[df_validation.cluster == df_validation.prediction].count().features / len(df_validation)
    print("Accuracy: {}%".format(round(accuracy * 100, 2)))
    df_validation = df_validation.drop('prediction', 1)
    
    df = label_dataset(df)
        
    print(df.groupby('cluster').count())
    
    try:
        #df_mlp = pd.read_pickle('machine_labeled.pkl')
        df_ml = pd.concat([df, df_ml], axis=0, join='outer', ignore_index=True)
        df_ml = df_ml.drop_duplicates(subset='text', keep="first")
        df_ml.to_pickle('machine_labeled.pkl')
    except:
        df.to_pickle('machine_labeled.pkl')
    LM_final = len(df_ml[df_ml.cluster != -1])
try:
    cgroups = range(2)
    for cgroup in cgroups:
        print("Messages from group {}".format(cgroup))
        for message in df_train.groupby('cluster').get_group(cgroup).text.head(20):
            print(message)
        print("")
except:
    print("Only one group")
    
os.remove('df.pkl')
os.remove('vocabulary.pkl')   

end = time.time()
print("Total Run-Time:  {}".format(round((end - start)/60,2)))


df_validation['prediction'] = group_predict(df_validation)
accuracy = df_validation[df_validation.cluster == df_validation.prediction].count().features / len(df_validation)
print("Accuracy: {}%".format(round(accuracy * 100, 2)))

'''
df_labeled = pd.read_pickle('labeled.pkl')

print(df.groupby('cluster').count())
df_labeled = df_labeled[df_labeled.cluster != -1]
df['answers'] = df['cluster']
df['cluster'] = -1.0
df = relate_dfs(df, [df_labeled])
df = group_predict(df)
print(df.groupby('cluster').count())
print("Accuracy: {}%".format(round(df[df.cluster == df.answers].count().features / len(df) * 100, 2))) 
'''