#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 18:06:32 2021

@author: jaehoon cha
@email: chajaehoon79@gmail.com
"""
from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset
import os
import re
import random 
import h5py

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

class To1DTensor(object):
    def __call__(self, sample):
        return torch.from_numpy(sample)

def transform0(sample):
    sample = sample.astype(np.float32)
    return sample

def transform1(sample):
    sample = sample.astype(np.float32)/255.
    return sample
    
def transform2(sample):
    sample = sample.astype(np.float32)
    sample = 2*(sample-.5)
    return sample

def transform3(sample):
    sample = sample.astype(np.float32)/255.
    sample = 2*(sample-.5)
    return sample

    


class Spectra(Dataset):
    def __init__(self, path_dir, transform=None):

        self.data_path = os.path.join(path_dir, "FT_spectra_gap_SLME_v2.h5")
        with h5py.File(self.data_path, 'r') as h5file:
            self.material_names = np.array(h5file['material_names']).astype(str)
            self.Spectra = h5file['alphas/cm^-1'][:]
            self.direct_gaps_array = h5file['direct_gaps'][:]
            self.slmes_array = h5file['SLME'][:]
            self.maxs_array = h5file['maxs'][:]
            self.positions_array = h5file['positions'][:]
            self.n_locals_array = h5file['n_local_maximums'][:]
            self.slopes_array = h5file['slopes'][:]
        
     
        self.transform = transform

    def __len__(self):
        return len(self.Spectra)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
                
        idx2 = random.randint(0, self.__len__()-1)

        sample1 = self.Spectra[idx]
        sample1 = self.transform(sample1)
        sample1 = torch.from_numpy(sample1).unsqueeze(0)


        sample2 = self.Spectra[idx2]
        sample2 = self.transform(sample2)
        sample2 = torch.from_numpy(sample2).unsqueeze(0)
        
        slme = np.array([self.slmes_array[idx]])
        slme = torch.from_numpy(slme)

        maxs = np.array([self.maxs_array[idx]])
        maxs = torch.from_numpy(maxs)

        positions = np.array([self.positions_array[idx]])
        positions = torch.from_numpy(positions)
        
        n_locals = np.array([self.n_locals_array[idx]])
        n_locals = torch.from_numpy(n_locals)

        slopes = np.array([self.slopes_array[idx]])
        slopes = torch.from_numpy(slopes)


        sample = {'x1':sample1,
                  'x2':sample2,
                  'slme':slme,
                  'max':maxs,
                  'position':positions,
                  'n_local':n_locals,
                  'slope':slopes
                  }

            
        return sample
    


