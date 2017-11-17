# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 23:35:08 2017

@author: KRapes

Supervised learning
"""

import warnings
import numpy as np
import json_management
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  AdaBoostClassifier
import Gaussian_hyperpara_selection
import time
import random


def relate_dfs(df_features, df_cluster):
    for i, row in df_cluster.iterrows():
        if row.text in list(df_features.text):
            idx = df_features.index[df_features['text'] == row.text]
            df_features.set_value(idx, 'cluster', row.cluster)
    return df_features

def print_messages(df):
    try:
        for cgroup in range(2):
            print("Messages from Group {}".format(cgroup))
            for message in df.groupby('prediction').get_group(cgroup).text.head(8):
                print(message)
            print("")
    except:
        print("Only one Group")


def split_set(df, test_size):
    X_train, X_test, y_train, y_test = train_test_split(list(df.features),
                                                        list(df.cluster),
                                                        test_size=test_size)
    return X_train, X_test, y_train, y_test
    
def best_classifier(df, percent_saved):
  
    def run_clf(clf, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            clf.set_params(**kwargs)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
        return score
    
         
    X_train, X_test, y_train, y_test = split_set(df, 0.5)
    
    best = {'score': 0, 'parameters': {}}
    best_overall = {'score': 0 , 'clf': None}
    names = ["NerualNet", "DecisionTree", "AdaBoost", "GaussianProcess"]
    
    parameters_MLP = {"hidden_layer_sizes": [25, 500], "alpha": [0,1.0]}
    parameters_DT = {"max_depth": [3, 100], "min_samples_split": [2, 20],"min_samples_leaf": [1, 20]}
    parameters_Ada = {"n_estimators": [25, 100],"learning_rate": [0,1.0] }
    parameters_Gauss = {}
    
    parameters = [parameters_MLP, parameters_DT, parameters_Ada, parameters_Gauss]
    
    clfs = [MLPClassifier( max_iter=500),
            DecisionTreeClassifier(),
            AdaBoostClassifier(),
            GaussianProcessClassifier(1.0 * RBF(1.0))]
   
    for name, parameters, clf in zip(names, parameters, clfs):
        start = time.time()
        findings = {}
        best = {'score': 0}
        for _ in range(200):
            try:
                kwargs = Gaussian_hyperpara_selection.next_values(parameters, findings)
                # A random value is added to increase resolution and avoid lost combinations because of similar scores
                score = run_clf(clf, **kwargs) + random.random()/1000
                findings[score] = kwargs
                if score > best['score']:
                    best['score'] = score
                    best['parameters'] = findings[max(findings)]
                    best['clf'] = clf
                if (time.time() - start) > 60:
                    start = time.time()
                    break
            except:
                break
                                                                                         
        print("Best Score for the {} classifier:   {}".format(name, round(best['score'], 2)))
        if best['score'] > best_overall['score']:
            best_overall['score'] = best['score']
            best_overall['clf'] = best['clf']
            

    return best_overall['clf'], round(best_overall['score'], 2)

def best_pruning_percent(clf, validation_idx):
    def generate_df(validation_idx):
        df = json_management.prepare_df_labeled(percent)
        df = df.drop(validation_idx)
        df = df[df.cluster != -1]
        return df
    
    percents = [1.0, .9, .8, .7, .6, .5, .4, .3, .2]
    scores = []
    for percent in percents:
        df = generate_df(validation_idx)
        X_train, X_test, y_train, y_test = split_set(df, 0.5)
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
    percent = percents[np.argmax(scores)]
    df = generate_df(validation_idx)
    clf.fit(list(df.features), list(df.cluster))
    for p, s in zip(percents, scores):
        print("Using the Most Polarizing {}%:     {}".format(p*100, round(s,2)))
    return clf, percent

def predict_cluster(clf, df):
    predictions = clf.predict(list(df.features))
    df['prediction'] = predictions
    print_messages(df)
    return df



