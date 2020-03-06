#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 15:53:48 2019

@author: aasl
"""
import math
import numpy as np
from scipy.io.matlab import loadmat
from keras.utils import np_utils
from keras.utils import Sequence
from os.path import dirname, join
from keras.utils import data_utils
data_utils._SEQUENCE_COUNTER = 0

max_record_size = 650000
null_class  = 6
default_mask = np.array(range(null_class))

dataset_path = dirname(__file__)

DS1 = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
       '122', '124', '201', '203', '205', '207', '208', '209', '215', '220', 
       '223', '230'];
   
DS2 = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
       '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
       '233', '234']


class MITDBSequence(Sequence):
    
    def __init__(self, 
                 data_set='DS1', 
                 vector_size=72, 
                 batch_size=500, 
                 hw_size=5, 
                 mask_class=default_mask,
                 use='all',
                 val_ratio=10.0
                 ):
        
        self.data_set = data_set
        self.vector_size = vector_size
        self.batch_size = batch_size
        self.hw_size = hw_size
        
        self.mask_class = mask_class
        self.null_class = np.max(mask_class)
        
        self.start_sample = 0
        self.record_size = max_record_size
        
        self.use = use
    
        if self.use == 'all':
            self.samples_per_record = max_record_size - vector_size
        elif self.use == 'train':
            self.record_size = int(np.floor(max_record_size * (1 - val_ratio/100)))
            self.samples_per_record =  self.record_size - vector_size
        else:
            self.record_size = int(np.floor(max_record_size * val_ratio / 100))
            self.samples_per_record = self.record_size - vector_size
            self.start_sample = int(np.floor(max_record_size * (1 - val_ratio/100)))
                
        data_x, y_val, y_pos, num_records = self.load_dataset(data_set)
        data_y = self.build_yval(y_val, y_pos, num_records, self.hw_size)
        
        self.num_records = num_records
        
        self.all_starts = list()
        self.all_ends   = list()
        
        for i in range(self.num_records):
            self.all_starts.extend(list(range(i * self.record_size, i * self.record_size + self.samples_per_record + 1)))
            self.all_ends.extend(list(range(i * self.record_size + self.vector_size, (i + 1) * self.record_size + 1)))
        
        
        self.len = math.ceil(self.num_records * self.samples_per_record / self.batch_size)
                    
        self.x = data_x
        self.y = data_y
    
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        
       batch_start = idx * self.batch_size
       batch_end   = (idx + 1) * self.batch_size

       batch_x = np.array([self.x[x:y] for (x,y) in zip(self.all_starts[batch_start:batch_end], 
                           self.all_ends[batch_start:batch_end])])

       batch_x = batch_x.reshape(batch_x.shape[0], self.vector_size, 1)     
    
       ibatch_y = np.array(self.y[self.all_starts[batch_start:batch_end]])
       
       batch_y = np_utils.to_categorical(ibatch_y, self.null_class+1)
       
       return (batch_x, batch_y.astype('float32'))
    
    
    def load_dataset(self, files='DS1'):
    
        if files == 'DS1' or files == 'train':
            num_records = 22
            DS = DS1
        elif files == 'DS2' or files == 'test':
            num_records = 22
            DS = DS2
        else:
            num_records = 1
            DS = [files]

        
        data_x = np.zeros(self.record_size * num_records, dtype='float32')
        data_y = list()
        data_z = list()
        
        for (filename, j) in zip(DS, range(num_records)):
            matfile = loadmat(join(dirname(__file__),'slmit_200.mat'))
            
            xtemp = np.array(matfile['s'], dtype='float32').squeeze()
            ytemp = np.array(matfile['trg']).squeeze() - 1
            ztemp = np.array(matfile['peakpos']).squeeze() - 1
            
            if self.use == 'train':
                xtemp = np.array(xtemp[self.start_sample:self.record_size])
                ytemp = np.array(ytemp[ztemp < self.record_size])
                ztemp = np.array(ztemp[ztemp < self.record_size])
            elif self.use != 'all':
                xtemp = np.array(xtemp[self.start_sample:self.start_sample + self.record_size])
                ytemp = np.array(ytemp[ztemp >= self.start_sample])
                ztemp = np.array(ztemp[ztemp >= self.start_sample])
            
            data_x[j * self.record_size: (j+1) * self.record_size] = xtemp
            data_y.extend(ytemp)
            data_z.extend(ztemp + j * self.record_size)
        
        return data_x, np.array(data_y), np.array(data_z), num_records


    def build_yval(self, y_val, y_pos, num_records=22, hw_size=5):
        
        offset = self.vector_size >>  1
        data_y = (self.null_class) * np.ones(self.record_size * num_records, dtype='uint8')
        
        for (i, j) in zip(y_pos, range(len(y_val))):
            data_y[i-hw_size-offset:i+hw_size-offset] = self.mask_class[y_val[j]] 
        
        return data_y
    
    