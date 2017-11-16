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
import os.path
import pickle
#import visuals as vs

json_location = "D:\Intelligens\challenge_en.json"

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)



def preprocess_data():
    def load_json():
        with open(json_location, 'r') as json_data:
            json_lines = []
            for i,line in enumerate(json_data):
                #if i >= 3000:
                #   break
                json_lines.append(json.loads(line))
           
        return pd.DataFrame.from_dict(json_lines)

    
    def label_features(df, features_master, remove_dpls=False):
        dupl = 0
        for i,row in df.iterrows():
            message = pruning_dict.remove_nonalphanumeric(row.text)
            features = Counter(message.split()) & features_master
            features = features + features_master
            features = list(np.array(list(features.values())) - 1)
            if remove_dpls == True and features in list(df.features):
                dupl += 1
                df.set_value(i,'features', None)
                df = df[pd.notnull(df['features'])]
            else:
                df.set_value(i,'features',features)

        return df, dupl
    

    try:
        df = load_obj('df')
        print("Loading df...")
    except:
        df = load_json()
        df = df[['text']]
        print("Len before drop {}".format(len(df)))
        df.drop_duplicates(['text'], keep='first')
        print("Len after drop {}".format(len(df)))
        vocabulary = pruning_dict.build_vocabulary(df.text)
        '''
        features_master = Counter(list(vocabulary.keys()))
        df["features"] = [[0] * len(vocabulary)] * len(df)
        df, dupl = label_features(df, features_master, True)
        '''
        save_obj(df, 'df' )
    
    print("Original message count {}".format(len(df)))
    
    try:
        vocabulary = load_obj('vocabulary')
        print("Loading Vocabulary...")
    except:
        vocabulary = pruning_dict.build_vocabulary(df.text)
        save_obj(vocabulary, 'vocabulary' )
    vocabulary = pruning_dict.prune_vocab(vocabulary)
    features_master = Counter(list(vocabulary.keys()))
    df["features"] = [[0] * len(vocabulary)] * len(df)
    df, dupl = label_features(df, features_master, False)
    
    return df, features_master






def prepare_df_labeled():
    df, features_master = preprocess_data()
    df["cluster"] = -1.0
    df_labeled = pd.read_pickle('labeled.pkl')
    
    for i, row in df_labeled.iterrows():
        if row.text in list(df.text):
            idx = df.index[df['text'] == row.text]
            df.set_value(idx[0],'cluster', row.cluster)
    return df












        
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
            for message in df.groupby('cluster').get_group(cgroup).text.head(10):
                print(message)
            print("")
    





