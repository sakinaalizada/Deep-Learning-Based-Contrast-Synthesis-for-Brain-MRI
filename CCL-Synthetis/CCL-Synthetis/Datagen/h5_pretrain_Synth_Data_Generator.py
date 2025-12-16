"""
@author: lavanya
Data generator for pretraining with MR-contrast guided contrastive learning approach
Given location of HDF5 files for images and the corresponding constraint maps, the datagenerator randomly selects images for training
"""

import sys
import tensorflow as tf
import numpy as np
from skimage.util import view_as_blocks
import h5py
import threading
import tensorflow as tf
import elasticdeform 

 
class DataLoaderObj(tf.keras.utils.Sequence):
    ''' Config cfg contains the parameters that control training'''
    def __init__(self, cfg, train_flag=True): 
        '''Edited by Parisima'''
        self.data_aug = cfg.data_aug       
        self.batch_size = cfg.batch_size
        self.contrast_idx = cfg.contrast_idx
        self.target_contrast_idx = cfg.target_contrast_idx
        self.num_channels = len(self.contrast_idx)
        self.num_output_channels= len(self.target_contrast_idx)
        self.cfg = cfg
        self.train_flag = train_flag
        if train_flag:
            print('Initializing training dataloader')
            self.input_hdf5_img     = h5py.File(cfg.hdf5_train_img_filename, 'r')
        else:
            print('Initializing validation dataloader')
            self.input_hdf5_img     = h5py.File(cfg.hdf5_val_img_filename, 'r')
           
        self.img = self.input_hdf5_img['img']

        # if train_flag:
        #     print("⚠️ Training HDF5 not found — generating dummy data")
        #     self.img = np.random.rand(40, cfg.img_size_x, cfg.img_size_y, len(cfg.contrast_idx) + len(cfg.target_contrast_idx))
        # else:
        #     print("⚠️ Validation HDF5 not found — generating dummy data")
        #     self.img = np.random.rand(20, cfg.img_size_x, cfg.img_size_y, len(cfg.contrast_idx) + len(cfg.target_contrast_idx))

       
    
        self.len_of_data = self.img.shape[0]
        self.num_samples = self.len_of_data // cfg.control_size_of_dataset  # use it to control size of samples per epoch
        print('Total samples :', self.num_samples)
        self.img_size_x = self.img.shape[1]
        self.img_size_y = self.img.shape[2]
        self.arr_indexes = np.random.choice(self.len_of_data, self.num_samples, replace=True)

        
    def __del__(self):
        # self.input_hdf5_img.close() 
        if hasattr(self, "input_hdf5_img"):
            self.input_hdf5_img.close()
 
        
    def __len__(self):
        return self.num_samples // self.batch_size

    def get_len(self):
        return self.num_samples // self.batch_size
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.arr_indexes = np.random.choice(self.len_of_data, self.num_samples, replace=True)
        
        
    def __shape__(self):
        data = self.__getitem__(0)
        return data.shape
    
    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        with threading.Lock():
            i = self.arr_indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
            x_train = self.generate_X(i)
            y_train = self.generate_Y(i)
            
        if self.data_aug and self.train_flag:  # Only augment if flag is on and it's the training phase
            x_train, y_train = self.generate_XY_aug(x_train, y_train)

        return tf.identity(x_train), tf.identity(y_train)
        
    def generate_X(self, list_idx):    
        X = np.zeros((self.batch_size, self.img_size_x, self.img_size_y, self.num_channels),dtype="float64")
        for jj in range(0, self.batch_size):                
            X[jj] = self.img[list_idx[jj],...,self.contrast_idx] 
            # X[jj] = np.transpose(self.img[list_idx[jj], ..., self.contrast_idx], (1, 2, 0))
        return X
    
    '''Edited by Parisima'''
    def generate_Y(self, list_idx):    
        Y = np.zeros((self.batch_size, self.img_size_x, self.img_size_y, self.num_output_channels),dtype="float64")
        for jj in range(0, self.batch_size):                
            Y[jj] = self.img[list_idx[jj],...,self.target_contrast_idx] 
            # Y[jj] = np.transpose(self.img[list_idx[jj], ..., self.target_contrast_idx], (1, 2, 0))
        return Y
    
    def generate_XY_aug(self, x_train, y_train):
        # Decide how many samples in the batch to augment
        num_to_augment = int(self.batch_size * self.cfg.augmentation_ratio)  # e.g., 0.5 for 50%
        aug_indices = np.random.choice(self.batch_size, num_to_augment, replace=False)
        x_train_aug = x_train.copy()
        y_train_aug = y_train.copy()
        for idx in aug_indices:
            # Perform the elastic transformation on both X and Y
            # Assume X and Y are both 4D: batch_size, width, height, channels
            # The transformation should be the same for all channels, hence performed on the entire array
            min_val_x = x_train[idx].min()
            max_val_x = x_train[idx].max()
            min_val_y = y_train[idx].min()
            max_val_y = y_train[idx].max()
            [x_temp, y_temp] = elasticdeform.deform_random_grid([x_train_aug[idx], y_train_aug[idx]], sigma = 3, points =5, order= 3, mode= 'reflect', cval = [min_val_x,min_val_y], axis = (0,1))
            x_train_aug[idx] = np.clip(x_temp, min_val_x, max_val_x)
            y_train_aug[idx] = np.clip(y_temp, min_val_y, max_val_y)

        return x_train_aug, y_train_aug