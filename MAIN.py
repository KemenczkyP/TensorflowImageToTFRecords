# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:38:13 2019

@author: KemyPeti
"""
import sys
import os
sys.path.append(os.getcwd() + "\\code\\TFRecordCreator")

import numpy as np
import tensorflow as tf

import TFFuncLib as TFEC

#%% write TFRecord with image
data_repos = '.\\numpy_data\\'

TFRGenerator = TFEC.TFRecordGenerator.generate(path = "TFRecord_container",
                                               filename = "2019_06_13_try")
for idx in range(10):
    #--------------------------------EXAMPLE INPUT----------------------------#
    A = np.array([idx*1.0, idx*2.0], dtype = np.float32) #the decode type have to be the same!!!
    np.save(data_repos + 'example_image' + str(idx) + '.npy', A)
    #--------------------------------EXAMPLE END------------------------------#
    
    #--------------------CREATE EXAMPLE FROM NUMPY ARRAY----------------------#
    feature_dict = TFEC.ImageToTfFeature.create(data_repos + 'example_image' + str(idx) + '.npy')
    
    feature_dict = TFEC.AddFeatureToDict(feature_dict = feature_dict,
                                         data_to_add_key = "label",
                                         data_to_add_value = 0.22*idx, #example_value
                                         type_ = "float")
    example = TFEC.FeatureDict2TfExample(feature_dict)
    #The feature_dict (and example) contains 2+dimension_number fields about the image:
    #   'image/encoded' - bit encoded image data
    #   'image/dim_size' - number of dimensions (for example it is 1)
    #   'image/1D_' - size of the first dimension
    #   'image/2D_' - size of the second dimension (if exists)
    #   'image/3D_' - size of the 3rd dimension (if exists)
    #   'image/4D_' - size of the 4th dimension (if exists)
    #   ...
    
    
    
    #-----------------------------WRITE TFRECORDS-----------------------------#
    #own function (it saves the tfrecord file and a pickle that contains the 
    #tfrecord informations for read)
    TFRGenerator.write(example, feature_dict)

TFRGenerator.close()


#%%
#--------------------------------READ TFRECORD DATASET------------------------#
TFRecReader = TFEC.TFRecordReader.read_keys(path = "TFRecord_container",
                                            filename = "2019_06_13_try")

dataset = TFRecReader.get_dataset()

#------------------------CALL THE EXAMPLES FROM THE DATASET-------------------#
#(NO ITERATOR ANYMORE!)
dataset = dataset.repeat(100)         #epoch_size
dataset = dataset.shuffle(100)    #shuffle_buffer_size
dataset = dataset.batch(5)          #batch size
dataset = dataset.prefetch(1)       #buffer_size

for serialized_examples in dataset:
    parsed = TFRecReader.parse_examples(serialized_examples)
    image =  tf.io.decode_raw(parsed['image/encoded'],
                              out_type=tf.float32, #the decode type have to be the same as the input type!!!
                              little_endian=True)
    