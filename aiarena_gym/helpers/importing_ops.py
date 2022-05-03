#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 14:12:12 2022

@author: brandinho
"""

import tensorflow as tf
import json

activation_functions = {
    "relu": tf.nn.relu,
    "elu": tf.nn.elu,
    "leaky_relu": tf.nn.leaky_relu,    
    "sigmoid": tf.nn.sigmoid,
    "tanh": tf.nn.tanh,
    "softmax": tf.nn.softmax
}

def load_tensorflow_model(path, file):
    imported_file = open(path + file, 'r')
    imported_model_dict = json.load(imported_file)
    imported_model_dict["activation_function"] = activation_functions[imported_model_dict["activation_function"]]
    return imported_model_dict