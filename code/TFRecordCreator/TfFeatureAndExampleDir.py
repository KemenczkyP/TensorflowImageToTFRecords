# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:50:06 2019

@author: KemyPeti
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if isinstance(value, six.string_types):                     
        value = six.binary_type(value, encoding='utf-8') 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def AddFeatureToDict(feature_dict,
                     data_to_add_key,
                     data_to_add_value,
                     type_ = "int"):
    '''
    Args:
        \n\t feature_dict : input feature dict
        \n\t data_to_add_key : key of the data in dict
        \n\t data_to_add_value : new data
        \n\t type_ : "int" for int type / "byte" for bytes / "float" for float
    Returns:
        \n\t feature_dict : appended dict
        
    '''
    if(type_ == "int"):
        feature_creator = _int64_feature
    elif(type_ == "byte"):
        feature_creator = _bytes_feature
    elif(type_ == "float"):
        feature_creator = _float_feature
    else:
        raise Exception("Incorrect arg: type_")
    
    try:
        feature_dict[data_to_add_key] = feature_creator(data_to_add_value)
    except:
        raise Exception("arg type_ and the type of arg data_to_add_value are not consistent")
    return feature_dict

def FeatureDict2TfExample(feature_dict):
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example