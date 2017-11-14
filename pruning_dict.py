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
            message_split = message.split()
            gram_count = 2
            grams = []
            for g in range(gram_count):
                for i in range(len(message_split) - 1):
                    gram = ''
                    for n in range(g):
                        gram = gram + message_split[i+n] + ' '
                    grams.append(gram)
            vocabulary = vocabulary + Counter(message_split) + Counter(grams)
        if word_drop == True:
            for key, count in dropwhile(
                                        lambda key_count: key_count[1] >= (len(messages) * .01),
                                        vocabulary.most_common()):
                del vocabulary[key]
        return vocabulary

def remove_nonalphanumeric(message):
    try:        
        message = message.lower()
        delchar_table = {ord(c): None for c in message if c not in 'abcdefghijklmnopqrstuvwxyz0123456789 '}
        return message.translate(delchar_table)
    except:
        print(message)
        return message
    
def prune_vocab(vocabulary):
    #try:
    labels = [ 'labeled.pkl']
    df = pd.DataFrame()
    for l in labels:
        df_temp = pd.read_pickle(l)
        df = pd.concat([df_temp, df], axis=0, join='outer', ignore_index=True)
        df = df.drop_duplicates(subset='text', keep="first")
    df = df.sort_values(by=['cluster'], ascending=False)
    df =  df[df.cluster != -1]
    df = df[pd.notnull(df['text'])]
    df = df.reset_index(drop=True)

    
    info = df.groupby('cluster').get_group(0)
    info_v = build_vocabulary(info.text, word_drop=True)
    info_w = info_v.keys()
    express = df.groupby('cluster').get_group(1)
    express_v = build_vocabulary(express.text, word_drop=True)
    express_w = express_v.keys()
    
    common_v = info_v & express_v
    common_w = common_v.keys()
    
    words = []
    ratios = []
    
    for word in info_w:
        if word not in common_w:
            ratios.append(info_v[word])
            words.append(word)
    
    for word in common_w:
        ratios.append(info_v[word] / express_v[word])
        words.append(word)
    
    for word in express_w:
        if word not in common_w:
            ratios.append(express_v[word] * -1)
            words.append(word)
    

    threshold = int(len(words) * .15)
    top20 = np.argsort(ratios)[-threshold:]
    bottom20 = np.argsort(ratios)[:threshold]

    polar_words = []
    for group in [top20, bottom20]:
        for index in group:
            polar_words.append(words[index])
    
    
    vocabulary = Counter(polar_words) & vocabulary
    #print(Counter(polar_words))
    #print("Length of polar_words {}".format(len(polar_words)))
    return vocabulary
    #except:
       # print("pruning error")
       # return vocabulary


