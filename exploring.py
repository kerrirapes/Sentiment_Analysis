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
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import visuals as vs

json_location = "D:\Intelligens\challenge_en.json"


def load_json(json_location):
    with open(json_location, 'r') as json_data:
        json_lines = []
        for i,line in enumerate(json_data):
            if i > 1000:
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

def label_features(df):
    for i,row in df.iterrows():
        message = remove_nonalphanumeric(row.text)
        features = Counter(message.split()) & features_master
        features = features + features_master
        features = np.array(list(features.values())) - 1
        df.set_value(i,'features',features)

def cluster_search(df2):
    for n in range(2,16):
        clusterer = GaussianMixture(n_components=n)
        clusterer.fit(df2)
        preds = clusterer.predict(df2)
        centers = clusterer.means_
        score = silhouette_score(df2,preds)
        print("The score for {} n_components in GaussianMixture is {}".format(n,score))
        print("")
    for n in range(2,16): 
        clusterer = KMeans(n_clusters=n)
        clusterer.fit(df2)
        preds = clusterer.predict(df2)
        centers = clusterer.cluster_centers_
        score = silhouette_score(df2,preds)
        print("The score for {} n_cluster in KMeans is {}".format(n,score))   


df = load_json(json_location)
df = df[['text']]

vocabulary = build_vocabulary(df.text)
features_master = Counter(list(vocabulary.keys()))

df["features"] = [[0] * len(vocabulary)] * len(df)

label_features(df)

#print(list(df.features))

df2 = pd.DataFrame(list(df.features), columns=range(len(features_master)))
pca = PCA(n_components=5)
pca.fit(df2)

#pca_samples = pca.transform(log_samples)
print(len(df2))
print("Explained_variance_ratio: {} ".format(pca.explained_variance_ratio_))
df2 = pca.transform(df2)
print(len(df2))
# Generate PCA results plot
#pca_results = vs.pca_results(df2, pca)
#pca_results.cumsum()

search = False
if search == True:
    cluster_search(df2)

N = 4
clusterer = KMeans(n_clusters=N)
clusterer.fit(df2)

# Predict the cluster for each data point
preds = clusterer.predict(df2)
df["cluster"] = preds
# Find the cluster centers
centers = clusterer.cluster_centers_

# Calculate the mean silhouette coefficient for the number of clusters chosen
score = silhouette_score(df2,preds)
print("The score for {} n_cluster in KMeans is {}".format(2,score))
print("")
print(df.groupby('cluster').count())
print("")

cgroups = range(N)
for cgroup in cgroups:
    print("Messages from group {}".format(cgroup))
    for message in df.groupby('cluster').get_group(cgroup).text.head(20):
        print(message)
    print("")
    




print("Done")

