# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import json
from collections import Counter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import pruning_dict
import os.path
import pickle


json_location = "D:\Intelligens\challenge_en.json"

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

def label_features(df, features_master):
        for i,row in df.iterrows():
            message = pruning_dict.remove_nonalphanumeric(row.text)
            features = Counter(message.split()) & features_master
            features = features + features_master
            features = list(np.array(list(features.values())) - 1)
            df.set_value(i,'features',features)
        return df

def cluster_filter(df, df2, N):
    clusterer = KMeans(n_clusters=N)
    clusterer.fit(df2)
    transform = clusterer.transform(df2)
    df['d_from_center'] = [min(x)**2 for x in transform]
    df['cluster'] = [np.argmin(x) for x in transform]
    for cgroup in range(N):
        group = df.groupby('cluster').get_group(cgroup)
        sum_squares = group.d_from_center.sum()
        mcount = group.d_from_center.count()
        std = ((sum_squares / (N-1))**0.5) / mcount
        if std < 0.01:
            df = df.drop(group.index)
    return df
        
def cluster_search(df2):
    def score_n(score, best_score, best_n):
        if score > best_score:
            best_score = score
            best_n = n
        return best_score, best_n
    search_range = min(50, len(df2))
    best_score = 0
    best_n = 1
    for n in range(2,search_range): 
        clusterer = KMeans(n_clusters=n)
        clusterer.fit(df2)
        preds = clusterer.predict(df2)
        try:
            score = silhouette_score(df2,preds)
            best_score, best_n = score_n(score, best_score, best_n)
        except:
            pass   
    return best_n

def create_feature_dataframe(df, features_master):
    return pd.DataFrame(list(df.features), columns=range(len(features_master)))

def filter_repeat(df, percent_saved):
    vocabulary = pruning_dict.build_vocabulary(df.text)
    vocabulary = pruning_dict.prune_vocab(vocabulary, percent_saved)
    features_master = Counter(list(vocabulary.keys()))
    df["features"] = [[0] * len(vocabulary)] * len(df)
    df = label_features(df, features_master)
    df2 = create_feature_dataframe(df, features_master)
    N = cluster_search(df2)
    df = cluster_filter(df, df2, N)
    df = df.drop_duplicates(['text'], keep='first')
    return df

def preprocess_data(percent_saved):
    def load_json():
        with open(json_location, 'r') as json_data:
            json_lines = []
            for i,line in enumerate(json_data):
                if i >= 150:
                   break
                json_lines.append(json.loads(line))
           
        return pd.DataFrame.from_dict(json_lines)

    try:
        df = load_obj('df')
    except:
        df = load_json()
        df = df[['text']]
        df = filter_repeat(df, 0.7)
        save_obj(df, 'df' )
        
    try:
        vocabulary = load_obj('vocabulary')
    except:
        vocabulary = pruning_dict.build_vocabulary(df.text)
        vocabulary = pruning_dict.prune_vocab(vocabulary, percent_saved)
        save_obj(vocabulary, 'vocabulary' )
    
    features_master = Counter(list(vocabulary.keys()))
    df["features"] = [[0] * len(vocabulary)] * len(df)
    df = label_features(df, features_master)
    
    return df, features_master

def prepare_df_labeled(percent_saved):
    try:
        os.remove('vocabulary.pkl')
    except:
        pass
    df, features_master = preprocess_data(percent_saved)
    df["cluster"] = -1.0
    df_labeled = pd.read_pickle('labeled.pkl')
    
    for i, row in df_labeled.iterrows():
        if row.text in list(df.text):
            idx = df.index[df['text'] == row.text]
            df.set_value(idx[0],'cluster', row.cluster)
    return df











'''
def pca_components(df, features_master):
    df2 = create_feature_dataframe(df, features_master)
    pca = PCA(n_components=10)
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
            for message in df.groupby('cluster').get_group(cgroup).text.head(3):
                print(message)
            print("")
        
'''
