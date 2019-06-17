# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:51:50 2019

@author: KemyPeti
"""

#%%

import tensorflow as tf

from ImageToTfFeature import ImageToTfFeature
from TfFeatureAndExampleDir import AddFeatureToDict, FeatureDict2TfExample

'''

feature_dict = ImageToTfFeature.create('something.npy')

feature_dict = AddFeatureToDict(feature_dict = feature_dict,
                                data_to_add_key = "label",
                                data_to_add_value = 0.22,
                                type_ = "float")
example = FeatureDict2TfExample(feature_dict)

'''
#%%

class TFRecordGenerator(object):
    
    def __init__(self):
        super.__init__()

    @classmethod
    def generate(self, path, filename):
        '''
        Generates a TFRecord file created in "path directory" with file name "filename".
        Args:
            \n\t path: The directory path 
            \n\t filename: name of the TFRecord file
            
        '''
        import os
        
        self.example_key = None
        self.example_types = []
        self.example_default_shape = []
        self.example_default_values = []
        self.output_file = os.path.join(path, filename)
        self.TFRecord_file = tf.io.TFRecordWriter(self.output_file + "_tfrecord" + ".tfrecord")
        '''
        TFRec_file = generate(path = "",
                              filename = "something_TFR")
        '''
        return self
    
    @classmethod
    def write(self, example, feature_dict):
        '''
        Writes an example into the TFRecord file generated before
        Args:
            \n\t example : tf.train.Example object
            \n\t feature_dict : feature_dict used for creating the tf.train.Example object
        '''
        if(self.example_key == None):
            self.example_key = list(feature_dict.keys())
            for idx in range(len(self.example_key)):
                if(len(feature_dict[self.example_key[idx]].float_list.value)>0):
                    self.example_default_values.append(-1)
                    self.example_default_shape.append([])
                    self.example_types.append(tf.float32)
                elif(len(feature_dict[self.example_key[idx]].int64_list.value)>0):
                    self.example_default_values.append(-1)
                    self.example_default_shape.append([])
                    self.example_types.append(tf.int64)
                elif(len(feature_dict[self.example_key[idx]].bytes_list.value)>0):
                    self.example_default_values.append('')
                    self.example_default_shape.append(())
                    self.example_types.append(tf.string)
                else:
                    self.example_default_values.append('')
                    self.example_default_shape.append([])
                    self.example_types.append(map())
                
             
        else:
            if(list(feature_dict.keys()) != self.example_key):
                raise Exception('Example key are not the same as before!')
            
        self.TFRecord_file.write(example.SerializeToString())
        
    @classmethod
    def close(self):
        '''
        Closes the TFRecord file and save the dict keys
        '''
        import pickle
        
        self.TFRecord_file.close()
        
        d = dict()
        d['key'] = self.example_key
        d['function'] = self.example_types
        d['init_val'] = self.example_default_values
        d['shape'] = self.example_default_shape
        
        
        pickle.dump(d, open(self.output_file + "_example_keys" + ".pickle", "wb" ) )

        
#%%
        
class TFRecordReader(object):
    
    def __init__(self):
        super.__init__()
    
    @classmethod
    def read_keys(self, path, filename):
        import pickle
        import os
        
        self.output_file = os.path.join(path, filename)
        example_keys = pickle.load(open(self.output_file + "_example_keys.pickle", "rb" ) )
        
        self.e_keys = example_keys['key']
        self.e_shape = example_keys['shape']
        self.e_funs = example_keys['function']
        self.e_init_vals = example_keys['init_val']
        
        return self
        
    @classmethod
    def get_dataset(self):
        self.feature_description = dict()
        for idx in range(len(self.e_keys)):
            self.feature_description[self.e_keys[idx]] = tf.io.FixedLenFeature(self.e_shape[idx],
                                                                               dtype = self.e_funs[idx],
                                                                               default_value = self.e_init_vals[idx])
        
        self.dataset = tf.data.TFRecordDataset([self.output_file + "_tfrecord.tfrecord"])
        
        return self.dataset
    
    @classmethod
    def parse_examples(self, serialized_examples):
        
        parsed_examples = tf.io.parse_example(serialized_examples,
                                              self.feature_description)
        return parsed_examples