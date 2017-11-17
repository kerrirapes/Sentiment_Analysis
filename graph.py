# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 22:31:53 2017

@author: KRapes
"""


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA



def create_feature_dataframe(df):
    l = len(df.iloc[0]['features'])
    return pd.DataFrame(list(df.features), columns=range(l))



def pca_components(df):
    df2 = create_feature_dataframe(df)
    pca = PCA(n_components=2)
    pca.fit(df2)
    print("Explained_variance_ratio: {} ".format(pca.explained_variance_ratio_))
    return pca.transform(df2)



def cluster_results(df):
  
    pca = pca_components(df.copy())
    df['Dimension 1'] = [x[0] for x in pca]
    df['Dimension 2'] = [x[1] for x in pca]

    #df = df[['pca0', 'pca1']]
    #predictions = pd.DataFrame(preds, columns = ['Cluster'])
    #plot_data = pd.concat([predictions, reduced_data], axis = 1)
    
    # Generate the cluster plot
    fig, ax = plt.subplots(figsize = (14,8))
    
    # Color map
    cmap = cm.get_cmap('gist_rainbow')
    
    # Color the points based on assigned cluster
    for i, cluster in df.groupby('cluster'):   
        cluster.plot(ax = ax, kind = 'scatter', x = 'Dimension 1', y = 'Dimension 2', \
                     color = cmap((i)*1.0/(2-1)), label = 'Cluster %i'%(i), s=30);
    
    # Plot centers with indicators
    '''
    for i, c in enumerate(centers):
        ax.scatter(x = c[0], y = c[1], color = 'white', edgecolors = 'black', \
                   alpha = 1, linewidth = 2, marker = 'o', s=200);
        ax.scatter(x = c[0], y = c[1], marker='$%d$'%(i), alpha = 1, s=100);
    '''
    
    
    # Set plot title
    ax.set_title("Cluster Learning on PCA-Reduced Data - Centroids Marked by Number\nTransformed Sample Data Marked by Black Cross");

'''
df = pd.DataFrame({'text': ["aaaaa", "bbbbb", 'ccccc'],
                   'features': [[1,1,1], [2,2,2], [3,3,3]],
                   'cluster': [1,0,1]})
print(df)
cluster_results(df, [[1,0,0], [2,1,1]])
'''