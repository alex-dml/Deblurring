# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 15:24:23 2025

@author: duminil
"""
from skimage.transform import resize
import numpy as np
from tensorflow.keras.utils import Sequence
import cv2
def _resize(img, resolution=480, padding=6):

    return resize(img, (resolution, resolution), preserve_range=True, mode='reflect', anti_aliasing=True )


class ImageGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self,
                 list_ID,
                 path_to_data, 
                 batch_size, 
                 shape_img, 
                 transformation=False,
                 shuffle = False
                 ):
        'Initialization'
        self.list_ID = list_ID
        self.path_to_data = path_to_data
        # self.gt_path = gt_path
        self.batch_size = batch_size
        self.shape_img = shape_img
        self.transformation = transformation
        self.shuffle = shuffle
     
        

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_ID) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Find list of IDs
        list_IDs_temp = [self.list_ID[k] for k in indexes]
        # print(list_IDs_temp)
        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_ID))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        blurry, clear = np.zeros( self.shape_img ), np.zeros( self.shape_img )

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
    
            x = cv2.imread(self.path_to_data + ID)/255.0
            
            blur = cv2.GaussianBlur(x,(7,7),cv2.BORDER_DEFAULT)

            
            blur =  _resize(blur, self.shape_img[1])
            x = _resize(x, self.shape_img[1])
            
            blurry[i] =  blur
            clear[i] = x

        return np.array(blurry), np.array(clear)