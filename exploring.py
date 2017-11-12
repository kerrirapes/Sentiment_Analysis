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
import pruning_dict
#import visuals as vs

json_location = "D:\Intelligens\challenge_en.json"

def preprocess_data():
    def load_json():
        with open(json_location, 'r') as json_data:
            json_lines = []
            for i,line in enumerate(json_data):
                if i >= 3000:
                   break
                json_lines.append(json.loads(line))
           
        return pd.DataFrame.from_dict(json_lines)

    
    def label_features(df, remove_dpls=False):
        dupl = 0
        for i,row in df.iterrows():
            message = pruning_dict.remove_nonalphanumeric(row.text)
            features = Counter(message.split()) & features_master
            features = features + features_master
            features = list(np.array(list(features.values())) - 1)
            if remove_dpls == True and features in list(df.features):
                dupl += 1
                df.set_value(i,'features', None)
            else:
                df.set_value(i,'features',features)
        return dupl
    
    
    df = load_json()
    df = df[['text']]
    print("Original message count {}".format(len(df)))
    dupl = 0
    for _ in range(3):
        
        vocabulary = pruning_dict.build_vocabulary(df.text)
        print("Original vocab size {}".format(len(vocabulary)))
        vocabulary = pruning_dict.prune_vocab(vocabulary)
        print("Original vocab size {}".format(len(vocabulary)))
        features_master = Counter(list(vocabulary.keys()))
        df["features"] = [[0] * len(vocabulary)] * len(df)
        remove_dpls = True if _ <= 2 or dupl > 0 else False
        dupl = label_features(df, remove_dpls)
        df = df[pd.notnull(df['features'])]
        print("Final message count {}".format(len(df)))
        
    return df, features_master








        
def cluster_search(df2):
    def score_n(score, best_score, best_n):
        if score > best_score:
            best_score = score
            best_n = n
        return best_score, best_n
    best_score = 0
    best_n = 1
    for n in range(2,16):
        clusterer = GaussianMixture(n_components=n)
        clusterer.fit(df2)
        preds = clusterer.predict(df2)
        #centers = clusterer.means_
        score = silhouette_score(df2,preds)
        print("The score for {} n_components in GaussianMixture is {}".format(n,score))
        best_score, best_n = score_n(score, best_score, best_n)
    for n in range(2,16): 
        clusterer = KMeans(n_clusters=n)
        clusterer.fit(df2)
        preds = clusterer.predict(df2)
        #centers = clusterer.cluster_centers_
        score = silhouette_score(df2,preds)
        print("The score for {} n_cluster in KMeans is {}".format(n,score))
        best_score, best_n = score_n(score, best_score, best_n)
    return best_n

def create_feature_dataframe(df, features_master):
    df2 = pd.DataFrame(list(df.features), columns=range(len(features_master)))
    return df2

def pca_components(df, features_master):
    df2 = create_feature_dataframe(df, features_master)
    pca = PCA(n_components=75)
    pca.fit(df2)
    print("Explained_variance_ratio: {} ".format(pca.explained_variance_ratio_))
    return pca.transform(df2)

def pca_explore(df, features_master):
    for pca_components in range(1,2):
        print("pca_components: {}".format(pca_components))
        
        #print(len(df2))
        # Generate PCA results plot
        #pca_results = vs.pca_results(df2, pca)
        #pca_results.cumsum()
        
        N = 2
        search = False
        if search == True:
            N = cluster_search(df2)
        
        
        clusterer = KMeans(n_clusters=N)
        clusterer.fit(df2)
        
        # Predict the cluster for each data point
        preds = clusterer.predict(df2)
        df["cluster"] = preds
        # Find the cluster centers
        centers = clusterer.cluster_centers_
        
        # Calculate the mean silhouette coefficient for the number of clusters chosen
        score = silhouette_score(df2,preds)
        print("The score for {} n_cluster in KMeans is {}".format(N,score))
        print("")
        print(df.groupby('cluster').count())
        print("")
        
        
        cgroups = range(N)
        for cgroup in cgroups:
            print("Messages from group {}".format(cgroup))
            for message in df.groupby('cluster').get_group(cgroup).text.head(10):
                print(message)
            print("")
    





