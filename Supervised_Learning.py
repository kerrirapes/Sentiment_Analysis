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


df, features_master = exploring.preprocess_data()
df["cluster"] = -1.0

df_labeled = pd.read_pickle('labeled.pkl')
#df = df_labeled

for i, row in df_labeled.iterrows():
    if row.text in list(df.text):
        idx = df.index[df['text'] == row.text]
        df.set_value(idx,'cluster', row.cluster)
        
df = df[df.cluster != -1]
#pca = exploring.pca_components(df, features_master)
#df['pca'] = list(pca)

X_train, X_test, y_train, y_test = train_test_split(list(df.features),
                                                    list(df.cluster),
                                                    test_size=0.33,
                                                    random_state=42)

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


start = time.time()
best = {'score': 0,
        'solver': 'lbfgs',
        'activation': 'relu',
        'parameters': {'hidden_layer_sizes': 25} }

parameters = {"hidden_layer_sizes": [25, 300]
              }
def run_clf(activation_kw, solver_kw, **kwargs):
    clf = MLPClassifier(alpha=1, max_iter=10000, activation=activation_kw, solver=solver_kw, **kwargs)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return score

for solver_kw in ['lbfgs', 'sgd', 'adam']: 
    for activation_kw in ['identity', 'logistic', 'tanh']:
        findings = {}
        for _ in range(100):
            kwargs = Gaussian_hyperpara_selection.next_values(parameters, findings)
            score = run_clf(activation_kw, solver_kw, **kwargs)
            findings[score] = kwargs
        print("Activation {} Solver {} Parameters {}".format(activation_kw, solver_kw,
                                                                                  findings[max(findings)]))
        print("Score:   {}".format(run_clf(activation_kw, solver_kw, **findings[max(findings)])))
        if score > best['score']:
            best['score'] = score
            best['solver'] = solver_kw
            best['activation'] =activation_kw
            best['parameters'] = findings[max(findings)]
        if (time.time() - start) > 120:
            start = time.time()
            print("BREAK FOR TIME")
            break
            
print("Activation {} Solver {} Parameters {}".format(best['activation'],
                                                                          best['solver'],
                                                                          best['parameters']))
                                                                                            
print("Best Score:   {}".format(run_clf(best['activation'], best['solver'], **best['parameters'])))

clf = GaussianProcessClassifier(1.0 * RBF(1.0))
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("Gaussian Classifier Score {}".format(score))