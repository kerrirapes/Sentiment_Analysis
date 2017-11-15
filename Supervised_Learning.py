# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 23:35:08 2017

@author: KRapes

Supervised learning
"""
import os
import pandas as pd
import numpy as np
import exploring
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import Gaussian_hyperpara_selection
import time
import random


df, features_master = exploring.preprocess_data()
df["cluster"] = -1.0

df_labeled = pd.read_pickle('labeled.pkl')
#df = df_labeled

for i, row in df_labeled.iterrows():
    if row.text in list(df.text):
        idx = df.index[df['text'] == row.text]
        df.set_value(idx[0],'cluster', row.cluster)
        
df = df[df.cluster != -1]
#pca = exploring.pca_components(df, features_master)
#df['pca'] = list(pca)

X_train, X_test, y_train, y_test = train_test_split(list(df.features),
                                                    list(df.cluster),
                                                    test_size=0.5)

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]
'''
score_sum = 0
best_score = 0
best_classifier = ""
for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        score_sum += score
        if score > best_score:
            best_score = score
            best_classifier = name
        print("{} scored {}".format(name, score))

print("")
print("The average score was {}".format(score_sum/len(names)))
print("The best classifier was {} with a score of {}".format(best_classifier, best_score))
'''

start = time.time()
best = {'score': 0,
        'solver': 'lbfgs',
        'activation': 'relu',
        'parameters': {'hidden_layer_sizes': 25} }

parameters = {"hidden_layer_sizes": [25, 500],
              "alpha": [0,1.0]
              }
def run_clf( **kwargs):
    clf = MLPClassifier( max_iter=500,  **kwargs)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return score

#for solver_kw in ['lbfgs', 'sgd', 'adam']: 
#    for activation_kw in ['identity',  'tanh']:
findings = {}
for _ in range(1): 
    kwargs = Gaussian_hyperpara_selection.next_values(parameters, findings)
    score = run_clf( **kwargs) + random.random()/1000
    findings[score] = kwargs

    if score > best['score']:
        best['score'] = score
        #best['solver'] = solver_kw
        #best['activation'] =activation_kw
        best['parameters'] = findings[max(findings)]
    if (time.time() - start) > 180:
        start = time.time()
        print("BREAK FOR TIME")
        break

   
                                                                                        
print("Best Score:   {}".format(run_clf(**best['parameters'])))
print(best['parameters'])


clf = GaussianProcessClassifier(1.0 * RBF(1.0))
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("Gaussian Classifier Score {}".format(score))


start = time.time()

best = {'parameters': {}, 'score': 0.0 }

parameters = {"max_depth": [3, 100],
              "min_samples_split": [2, 20],
              "min_samples_leaf": [1, 20],
              }

def run_clf( **kwargs):
    clf = DecisionTreeClassifier(**kwargs)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return score


findings = {}
for _ in range(10): 
    kwargs = Gaussian_hyperpara_selection.next_values(parameters, findings)
    score = run_clf( **kwargs) + random.random()/1000
    findings[score] = kwargs 

    if score > best['score']:
        best['score'] = score
        best['parameters'] = findings[max(findings)]
    if (time.time() - start) > 180:
        start = time.time()
        print("BREAK FOR TIME")
        break

                                                                                       
print("Best Score:   {}".format(run_clf( **best['parameters'])))
print(best['parameters'])
DT = best



best = {'parameters': {}, 'score': 0.0 }

parameters = {"n_estimators": [25, 100],
              "learning_rate": [0,1.0]
              }

base = DecisionTreeClassifier(**DT['parameters'])
def run_clf(base, **kwargs):
    clf = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=base,**kwargs)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return score


findings = {}
for _ in range(10): 
    kwargs = Gaussian_hyperpara_selection.next_values(parameters, findings)
    score = run_clf(base, **kwargs) + random.random()/1000
    findings[score] = kwargs 

    if score > best['score']:
        best['score'] = score
        best['parameters'] = findings[max(findings)]
    if (time.time() - start) > 180:
        start = time.time()
        print("BREAK FOR TIME")
        break
    
                                                                                       
print("Best Score:   {}".format(run_clf(base, **best['parameters'])))
print(best['parameters'])




#os.remove('df.pkl')
#os.remove('vocabulary.pkl')