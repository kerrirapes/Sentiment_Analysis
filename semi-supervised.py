# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 18:48:17 2017

@author: KRapes

Semi-Supervised
"""

import pandas as pd
import numpy as np
import exploring
from scipy import stats
from sklearn.semi_supervised import label_propagation

def keyboard_input(text):
    user_input = None
    while user_input not in ['i', 'e', ' ', 'q']:
        print(text)
        user_input = input("")
    if user_input == 'i':
        df.set_value(i,'cluster', 0)
    elif user_input == 'e':
        df.set_value(i,'cluster', 1)
    elif user_input == ' ':
        df.set_value(i,'cluster', -1)
    elif user_input == 'q':
        return
    else:
        print("Error")
    return


df, features_master = exploring.preprocess_data()
df["cluster"] = -1.0
#keyboard_input(df.iloc[0].text)
#df.to_pickle('labeled.pkl')
df_labeled = pd.read_pickle('labeled.pkl')
df_labeled.to_pickle('labeled_previous.pkl')
for i, row in df_labeled.iterrows():
    if row.text in list(df.text):
        idx = df.index[df['text'] == row.text]
        df.set_value(idx,'cluster', row.cluster)


pca = exploring.pca_components(df, features_master)
df['pca'] = list(pca)

max_iterations = 1



for i in range(max_iterations):
    df = df.sort_values(by=['cluster'], ascending=False)
    df = df.reset_index(drop=True)
    x = np.array(list(df.pca))
    y = np.array(list(df.cluster))
    n_total_samples = len(y)
    n_labeled_points = (df.cluster.values == -1).argmax()
    
    print(df.groupby('cluster').count())    
    
    if n_labeled_points == n_total_samples:
        print("No unlabeled items left to label.")
        break


    lp_model = label_propagation.LabelSpreading(gamma=0.25, max_iter=max_iterations)
    lp_model.fit(x, y)

    
    print("Iteration %i %s" % (i, 70 * "_"))
    print("Label Spreading model: %d labeled & %d unlabeled (%d total)"
          % (n_labeled_points, n_total_samples - n_labeled_points,
             n_total_samples))


    # Calculate uncertainty values for each transduced distribution
    pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)
    uncertainty_index = np.argsort(pred_entropies)
    
    labeled_count = 0
    for i in uncertainty_index:
        if df.iloc[i]['cluster'] == -1:
            labeled_count += 1
            print(pred_entropies[i], lp_model.predict([x[i]]))
            keyboard_input(df.iloc[i].text)
        if labeled_count >= 10:
            break

    


df.to_pickle('labeled.pkl')
predictions = lp_model.predict(x)
df['prediction'] = predictions

print(df.groupby('prediction').count())
print(df.groupby('cluster').count())
print("Score: {}".format(lp_model.score(x,y)))
print("")

try:
    cgroups = range(2)
    for cgroup in cgroups:
        print("Messages from group {}".format(cgroup))
        for message in df.groupby('prediction').get_group(cgroup).text.head(10):
            print(message)
        print("")
except:
    print("Only one group")