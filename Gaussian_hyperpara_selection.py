# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 16:07:52 2017

@author: KRapes

Gaussian Hyperparameter Selection
Based on scripts made of a tenserflow network for OCR
https://github.com/krapes/OCR_Receipt_Reading


"""

import random
import numpy as np
from sklearn.gaussian_process import GaussianProcess

RESOLUTION = 100.0

def hyperparam_wslist(scores, parameters, n_hidden_range):
    def fit_curve(parameters, scores, n_hidden_choices):
            y_mean, y_std = gaussian_process(parameters, scores, n_hidden_choices)
            y_max = max(scores)
            y_highestexpected = (y_mean + 1.96 * y_std)
            expected_improvement = y_highestexpected - y_max
            expected_improvement[expected_improvement < 0] = 0
            max_index = expected_improvement.argmax()
            return n_hidden_choices[max_index]
        
    def gaussian_process(x_train, y_train, x_test):
        def vector_2d(array):
            return np.array(array).reshape((-1, 1))
        
        import warnings
    
        
        
        def fxn():
            warnings.warn("deprecated", DeprecationWarning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fxn()
            
            x_train = vector_2d(x_train)
            y_train = vector_2d(y_train)
            x_test = vector_2d(x_test)
            # Train gaussian process
            gp = GaussianProcess(corr='squared_exponential',
                                 theta0=1e-1, thetaL=1e-3, thetaU=1)
            gp.fit(x_train, y_train)
    
            # Get mean and standard deviation for each possible
            # number of hidden units
            y_mean, y_var = gp.predict(x_test, eval_MSE=True)
            y_std = np.sqrt(vector_2d(y_var))
    
        return y_mean, y_std
    
    
    min_n_hidden, max_n_hidden = n_hidden_range
    if isinstance(min_n_hidden, int) and isinstance(max_n_hidden, int):
        n_hidden_choices = np.arange(min_n_hidden, max_n_hidden + 1)
        return int(fit_curve(parameters, scores, n_hidden_choices))
    else:
        step = (max_n_hidden - min_n_hidden) / RESOLUTION
        n_hidden_choices = [ min_n_hidden + + random.random()/1000 + step * x for x in range(int(RESOLUTION)) ]
        return fit_curve(parameters, scores, n_hidden_choices)


def findings_search(findings, hyparam_name):
    scores = []
    parameters = [] 
    sorted_dict = {}
    for score in findings:
        parameter = findings[score][hyparam_name]
        if parameter in sorted_dict:
            if score > sorted_dict[parameter]['score']:
                sorted_dict[parameter] = {'score': score}
        else:
            sorted_dict[parameter] = {'score': score}
            
    for parameter in sorted_dict:
        parameters.append(parameter)
        scores.append(sorted_dict[parameter]['score'])

    return scores, parameters


def elect_value(parameter, findings, valuerange):
    scores, parameters = findings_search(findings, parameter)
    if len(scores) >= 10:
        return hyperparam_wslist(scores, parameters, valuerange)
    else:
        if isinstance(valuerange[0], int) and isinstance(valuerange[1], int):
            return random.randint(valuerange[0],valuerange[1])
        else:
            return random.uniform(valuerange[0], valuerange[1])



def next_values(parameters, findings):
    parameter_values = {}
    for parameter in parameters:
        next_value = elect_value(parameter, findings, parameters[parameter])
        parameter_values[parameter] = next_value
    return parameter_values


    