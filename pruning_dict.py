# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 15:10:27 2017

@author: KRapes
"""
import pandas as pd
from collections import Counter
from itertools import dropwhile
import numpy as np

def build_vocabulary(messages, word_drop=True):
        vocabulary = Counter()
        for message in messages:
            message = remove_nonalphanumeric(message)
            vocabulary = vocabulary + Counter(message.split())
        if word_drop == True:
            for key, count in dropwhile(
                                        lambda key_count: key_count[1] >= (len(messages) * .01),
                                        vocabulary.most_common()):
                del vocabulary[key]
        return vocabulary

def remove_nonalphanumeric(message):
        message = message.lower()
        delchar_table = {ord(c): None for c in message if c not in 'abcdefghijklmnopqrstuvwxyz0123456789 '}
        return message.translate(delchar_table)
    
def prune_vocab(vocabulary):
    #try:
    df = pd.read_pickle('labeled.pkl')
    df = df.sort_values(by=['cluster'], ascending=False)
    df = df.reset_index(drop=True)
    
    info = df.groupby('cluster').get_group(0)
    info_v = build_vocabulary(info.text, word_drop=True)
    express = df.groupby('cluster').get_group(1)
    express_v = build_vocabulary(express.text, word_drop=True)
    common = info_v & express_v
    
    words = list(common.keys())
    ratios = []
    for word in list(words):
        ratios.append(info_v[word] / express_v[word])
    
    threshold = int(len(words) * .1)
    top20 = np.argsort(ratios)[-threshold:]
    bottom20 = np.argsort(ratios)[:threshold]

    polar_words = []
    for group in [top20, bottom20]:
        for index in group:
            polar_words.append(words[index])
    
    vocabulary = Counter(polar_words) & vocabulary
  
    print("Length of polar_words {}".format(len(polar_words)))
    return vocabulary
    #except:
       # print("pruning error")
       # return vocabulary


