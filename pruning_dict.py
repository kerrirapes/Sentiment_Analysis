# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 15:10:27 2017

@author: KRapes
"""
import pandas as pd
from collections import Counter
from itertools import dropwhile

def build_vocabulary(df, word_drop=True):
        vocabulary = Counter()
        for message in df:
            message = remove_nonalphanumeric(message)
            vocabulary = vocabulary + Counter(message.split())
        if word_drop == True:
            for key, count in dropwhile(lambda key_count: key_count[1] >= (len(df) * .01), vocabulary.most_common()):
                del vocabulary[key]
        return vocabulary

def remove_nonalphanumeric(message):
        message = message.lower()
        delchar_table = {ord(c): None for c in message if c not in 'abcdefghijklmnopqrstuvwxyz0123456789 '}
        return message.translate(delchar_table)
    
def prune_vocab(vocabulary):
    try:
        df = pd.read_pickle('labeled.pkl')
        df = df.sort_values(by=['cluster'], ascending=False)
        df = df.reset_index(drop=True)
        
        info = df.groupby('cluster').get_group(0)
        info_v = build_vocabulary(info.text, word_drop=True)
        express = df.groupby('cluster').get_group(1)
        express_v = build_vocabulary(express.text, word_drop=True)
        common = info_v & express_v
        
        common_words = []
        ratios = []
        for word in list(common.keys()):
            ratio = info_v[word] / express_v[word]
            if ratio <= 1.75 and ratio >= 0.25:
                common_words.append(word)
                ratios.append(ratio)
        for word in common_words:
            if word in vocabulary:
                del vocabulary[word]
        #print(dict(zip(common_words, ratios)), len(vocabulary))
        print("Length of common_words {}".format(len(common_words)))
        return vocabulary
    except:
        print("pruning error")
        return vocabulary


