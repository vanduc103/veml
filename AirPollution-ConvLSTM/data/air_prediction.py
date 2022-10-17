import gzip
import math
import numpy as np
import os
from PIL import Image
import random
import torch
import torch.utils.data as data
#from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import h5py

class MinMaxScaler:
    """
    Normalize the input
    """

    def __init__(self, minvalue, maxvalue):
        self.minvalue = minvalue
        self.maxvalue = maxvalue
        self.max_minus_min = self.maxvalue - self.minvalue

    def transform(self, data):
        return (data - self.minvalue) / self.max_minus_min

    def inverse_transform(self, data):
        return (data * self.max_minus_min) + self.minvalue

def load_data(filepath):
    # load pollution data
    pollution_file = filepath #'mydata/pollutionPM25.h5'
    X = []
    if os.path.isfile(pollution_file):
        with h5py.File(pollution_file, 'r') as hf:
            X = hf['pollution'][:]
            station_map = hf['station_map'][:]
            # normalize features
            scaler = MinMaxScaler(X.min(), X.max())
            #X = scaler.fit_transform(X.reshape(X.shape[0]*X.shape[1],1)).reshape(X.shape[0], X.shape[1])
            X = scaler.transform(X)
            print(sum(X[0]))
    return X, station_map, scaler

def load_fixed_set(root, is_train):
    # Load the fixed dataset
    filename = 'mnist_test_seq.npy'
    path = os.path.join(root, filename)
    dataset = np.load(path)
    dataset = dataset[..., np.newaxis]
    return dataset


class AirPrediction(data.Dataset):
    def __init__(self, dataset_type, filepath, timesteps,
                 transform=None):
        '''
        param num_objects: a list of number of possible objects.
        '''
        super(AirPrediction, self).__init__()

        X, station_map, scaler = load_data(filepath)
        
        # split to train, validate, test set
        train_size = (365+366)*24
        X_train, X_test = X[:train_size], X[train_size:]
        split_size = train_size - (92)*24
        X_train, X_val = X_train[:split_size], X_train[split_size:]
        print('Training set shape: {}'.format(X_train.shape))
        print('Validate set shape: {}'.format(X_val.shape))
        print('Test set shape: {}'.format(X_test.shape))
        if dataset_type == 1:
            self.dataset = X_train
        elif dataset_type == 2:
            self.dataset = X_val
        elif dataset_type == 3:
            self.dataset = X_test
        else:
            print('No support dataset type {}'.format(dataset_type))
        
        self.timesteps = timesteps
        self.pred_timesteps = timesteps
        self.length = self.dataset.shape[0] - self.timesteps - self.pred_timesteps
        self.image_size = 32
        self.in_channel = 1
        self.station_map = station_map
        self.loss_ratio = (self.image_size*self.image_size)/len(station_map)
        self.scaler = scaler

    def __getitem__(self, idx):
        offset = idx
        x = self.dataset[offset : offset + self.timesteps]
        y = self.dataset[offset + self.timesteps : offset + self.timesteps + self.pred_timesteps]
        y_mask = np.zeros_like(y)
        y_mask[:, self.station_map] = 1.0
        x = x.reshape((self.timesteps, self.in_channel, self.image_size, self.image_size))
        y = y.reshape((self.timesteps, self.in_channel, self.image_size, self.image_size))
        y_mask = y_mask.reshape((self.timesteps, self.in_channel, self.image_size, self.image_size))
        x = torch.from_numpy(x).contiguous().float()
        y = torch.from_numpy(y).contiguous().float()
        y_mask = torch.from_numpy(y_mask).contiguous().float()
        
        return [x, y, y_mask]

    def __len__(self):
        return self.length
