# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:29:52 2019

@author: KemyPeti
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import six


class ImageToTfFeature(object):
    def __init__(self):
        super.__init__()
        
    @classmethod
    def create(self,
               image_path):
        '''
        Args:
            \n\timage_path : path to .npy image
        Returns:
            \n\tdict of features which contains 
            \n\t\timage/encoded value : image buffer
            \n\t\timage/dim_size value : number of the image dimensions 
            (2 for Grayscale images, 3 for RGB images)
        '''
        self.__image_path = image_path
        
        image_buffer, ImageShape = self.__process_image(filename=self.__image_path) 
    
        feature_dict = self._ImageToTFFeature(self.__image_path,
                                             image_buffer,
                                             ImageShape)
        return feature_dict
    @classmethod
    def __process_image(self, filename):
        """Process a single image file.
        Args:
            filename: string, path to an image file in numpy format e.g., 'example.npy'
        Returns:
            image_buffer: string, encoded numpy array (should be 3D + a channel dimension)
            image_shape: list of ints 
        """
    
        # Read the numpy array.
        with tf.io.gfile.GFile(filename, 'rb') as f:
            image_data = f.read()
        
        try:
            #extract numpy header information
            header_len,dt,index_order,np_array_shape = self.__interpret_npy_header(image_data)
            image = np.frombuffer(image_data, dtype=dt, offset=10+header_len)
            image = np.reshape(image,np_array_shape,order=index_order)
            
        except ValueError as err:
                
                print(err)
        
        # Check that the image is a 3D array + a channel dimension
        self.__DimNum = len(image.shape)
        
        return image_data[10+header_len:], image.shape
    
    @classmethod
    def __interpret_npy_header(self,encoded_image_data):
        """Extracts numpy header information from byte encoded .npy files
        
        Args:
            encoded_image_data: string, (bytes) representation of a numpy array as the tf.gfile.FastGFile.read() method returns it
        Returns:
            header_len: integer, length of header information in bytes
            dt: numpy datatype with correct byte order (little or bigEndian)
            index_order: character 'C' for C-style indexing, 'F' for Fortran-style indexing
            np_array_shape: numpy array, original shape of the encoded numpy array
        """
        #Check if the encoded data is a numpy array or not
        numpy_prefix = b'\x93NUMPY'
        if encoded_image_data[:6] != numpy_prefix:
            raise ValueError('The encoded data is not a numpy array')
        
        if len(encoded_image_data)>10:
            header_len=np.frombuffer(encoded_image_data, dtype=np.uint16, offset=8,count=1)[0]
            header_data=str(encoded_image_data[10:10+header_len])
            
            dtypes_dict = {'u1': np.uint8, 'u2': np.uint16, 'u4' : np.uint32, 'u8': np.uint64,
                           'i1': np.int8, 'i2': np.int16, 'i4': np.int32, 'i8': np.int64,
                           'f4': np.float32, 'f8': np.float64, 'b1': np.bool}
    
            start_datatype = header_data.find("'descr': ")+10
            dt = dtypes_dict[header_data[start_datatype+1:start_datatype+3]]
            
            if (header_data[start_datatype:start_datatype+1] == '>'):
                dt = dt.newbyteorder('>')
            
            index_order='C'
            start_index_order=header_data.find("'fortran_order': ")+17
            if header_data[start_index_order:start_index_order+4]=='True':
                index_order='F'
            
            start_shape = header_data.find("'shape': (")+10
            end_shape = start_shape + header_data[start_shape:].find(")")
            
            np_array_shape=np.fromstring(header_data[start_shape:end_shape],dtype=int, sep=',')
            
            return header_len,dt,index_order,np_array_shape
        else:
            raise ValueError('The encoded data length is not sufficient')

    @classmethod
    def _ImageToTFFeature(self, filename, image_buffer, ImageShape):
        """Build an Example proto for an example.
        Args:
            filename: string, path to an image file, e.g., '/path/to/example.npy'
            image_buffer: string, encoded 3D numpy array
            ImageShape: integer, image size
        Returns:
            Example proto
        """
        
        feature = dict()
        feature['image/encoded'] = self._bytes_feature(image_buffer)
        feature['image/dim_size'] = self._int64_feature(len(ImageShape))
        
        for idx in range(self.__DimNum):
            feature['image/' + str(idx+1) + 'D_'] = self._int64_feature(ImageShape[idx])
        return feature

    @classmethod
    def _bytes_feature(self,value):
        """Wrapper for inserting bytes features into Example proto."""
        if isinstance(value, six.string_types):                     
            value = six.binary_type(value, encoding='utf-8') 
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    @classmethod
    def _int64_feature(self,value):
        """Wrapper for inserting int64 features into Example proto."""
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    
    @classmethod
    def _float_feature(value):
        """Wrapper for inserting float features into Example proto."""
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))


#feature_dict = ImageToTfFeature.create('something.npy')
        