#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 13:08:38 2024

@author: ubuntu
"""
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F



class Conv_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv_BN,self).__init__()
        self.layers =  nn.ModuleList()
        self.layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride = stride, padding = padding))
        self.layers.append(nn.BatchNorm1d(out_channels))

    def forward(self,x):   
        for idx in range(len(self.layers)):
            x = self.layers[idx](x)
        return  x
  
    
class Conv_LRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv_LRelu,self).__init__()
        self.layers =  nn.ModuleList()
        self.layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride = stride, padding = padding))
        self.layers.append(nn.LeakyReLU())

    def forward(self,x):   
        for idx in range(len(self.layers)):
            x = self.layers[idx](x)
        return  x
    
    

class Conv_BN_LRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias = True):
        super(Conv_BN_LRelu,self).__init__()
        self.layers =  nn.ModuleList()
        self.layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride = stride, padding = padding, bias = bias))
        self.layers.append(nn.BatchNorm1d(out_channels))
        self.layers.append(nn.LeakyReLU())
    
    def forward(self,x):   
        for idx in range(len(self.layers)):
            x = self.layers[idx](x)
        return  x


class ConvT_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super(ConvT_BN,self).__init__()
        self.layers =  nn.ModuleList()
        self.layers.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride = stride, \
                                              padding = padding, output_padding = output_padding))
        self.layers.append(nn.BatchNorm1d(out_channels))

    
    def forward(self,x):   
        for idx in range(len(self.layers)):
            x = self.layers[idx](x)
        return  x


class ConvT_BN_LRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ConvT_BN_LRelu,self).__init__()
        self.layers =  nn.ModuleList()
        self.layers.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride = stride, \
                                              padding = padding, output_padding = output_padding, bias = bias))
        self.layers.append(nn.BatchNorm1d(out_channels))
        self.layers.append(nn.LeakyReLU())
    
    def forward(self,x):   
        for idx in range(len(self.layers)):
            x = self.layers[idx](x)
        return  x



class Down_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias = True):
        super(Down_Block,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.CBR0 =  Conv_BN_LRelu(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = bias)
        self.CBR1 = Conv_BN_LRelu(out_channels, out_channels, kernel_size, stride = 1, padding = 'same', bias = bias)
        
        self.short_down = Conv_BN(in_channels, out_channels, kernel_size, stride = stride, padding = padding)
        
    
    def forward(self, x): 
        residual = x
        out = self.CBR0(x)           
        out = self.CBR1(out)
        if (self.in_channels != self.out_channels) or (self.stride > 1):
            residual = self.short_down(x)
        y = out + residual
        return y
    
    
    
class Up_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(Up_Block,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.CBR0 =  ConvT_BN_LRelu(in_channels, out_channels, kernel_size = kernel_size, stride = stride, \
                                   padding = padding, output_padding = output_padding, bias = bias)
        self.CBR1 = Conv_BN_LRelu(out_channels, out_channels, kernel_size, stride = 1, padding = 'same', bias = bias)
        
        self.short_up = ConvT_BN(in_channels, out_channels, kernel_size = kernel_size, stride = stride, \
                                   padding = padding, output_padding = output_padding)
        
    
    def forward(self, x): 
        residual = x
        out = self.CBR0(x)           
        out = self.CBR1(out)
        if (self.in_channels != self.out_channels) or (self.stride > 1):
            residual = self.short_up(x)
        y = out + residual
        return y
    

def stretch(X, alpha, gamma, beta, moving_mag, moving_min, eps, momentum, training):
    '''
    the code is based on the batch normalization in 
    http://preview.d2l.ai/d2l-en/master/chapter_convolutional-modern/batch-norm.html
    '''
    if not training:
        X_hat = (X - moving_min)/moving_mag
    else:
        assert len(X.shape) in (2, 4)
        min_ = X.min(dim=0)[0]
        max_ = X.max(dim=0)[0]

        mag_ = max_ - min_
        X_hat =  (X - min_)/mag_   
        moving_mag = momentum * moving_mag + (1.0 - momentum) * mag_
        moving_min = momentum * moving_min + (1.0 - momentum) * min_
    Y = (X_hat*gamma*alpha) + beta       
    return Y, moving_mag.data, moving_min.data




class Stretch(nn.Module):
    '''
    the code is based on the batch normalization in 
    http://preview.d2l.ai/d2l-en/master/chapter_convolutional-modern/batch-norm.html
    '''
    def __init__(self, num_features, num_dims, alpha):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.alpha = alpha
        self.gamma = nn.Parameter(0.01*torch.ones(shape))
        self.beta = nn.Parameter(np.pi*torch.ones(shape))
        self.register_buffer('moving_mag', 1.*torch.ones(shape))
        self.register_buffer('moving_min', np.pi*torch.ones(shape))

    def forward(self, X):
        if self.moving_mag.device != X.device:
            self.moving_mag = self.moving_mag.to(X.device)
            self.moving_min = self.moving_min.to(X.device)
        Y, self.moving_mag, self.moving_min = stretch(
            X, self.alpha.to(X.device) , self.gamma, self.beta, self.moving_mag, self.moving_min,
            eps=1e-5, momentum=0.99, training = self.training)
        return Y    


class SPC(nn.Module):
    def __init__(self, lat_dim):
        super(SPC, self).__init__()

        self.latent_dim = lat_dim

        self.encoder = nn.Sequential(
            Down_Block(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1), 
            Down_Block(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),  
            Down_Block(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),    
            Down_Block(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),  
        )

        self.to_mu = nn.Linear(64*3, self.latent_dim)
        self.to_dec = nn.Linear(self.latent_dim, 64*3)

        self.decoder = nn.Sequential(
            Up_Block(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding = 1),  
            Up_Block(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding = 0),  
            Up_Block(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding = 0),  
            Up_Block(in_channels=8, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding = 0), 
            nn.Conv1d(1, 1, 3, padding = 1),
            nn.Sigmoid()  
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample(self, num_samples = 100, z = None):
        if z is None:
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
        samples = self.decode(z)
        return samples

    def reconstr(self, x):
        return self.forward(x)[0]  

    def encode(self, x):     
        x = self.encoder(x) 
        x = torch.flatten(x, start_dim=1)   
        x = self.to_mu(x)
        return x


    def latent(self, x):
        z = self.encode(x)
        return z

    def decode(self, x):       
        x = nn.LeakyReLU()(self.to_dec(x))
        x = x.view(-1, 64, 3)
        x = self.decoder(x)
        return x

        
    def forward(self, x):
        z = self.encode(x)
        reconstr = self.decode(z)
        return [reconstr, z]  
    
    

class SPD(SPC):
    def __init__(self, lat_dim, alpha):
        super(SPD, self).__init__(lat_dim)
        self.alpha = alpha

        self.strecth = Stretch(self.latent_dim, 2, self.alpha)
        self.to_dec = nn.Linear(self.latent_dim*2, 64*3)

    def sample(self, num_samples = 100, z = None):
        if z is None:
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
        c = torch.cat((torch.cos(2*np.pi*z), torch.sin(2*np.pi*z)), 0)
        c = c.T.reshape(self.latent_dim*2, -1).T
        samples = self.decode(c)
        return samples


    def reconstr(self, x):
        z = self.encode(x)
        z2 = torch.cat((torch.cos(2*np.pi*z), torch.sin(2*np.pi*z)), 0)
        z2 = z2.T.reshape(self.latent_dim*2, -1).T
        reconstr = self.decode(z2)
        return reconstr   

    def encode(self, x):          
        x = self.encoder(x)  
        x = torch.flatten(x, start_dim=1)   
        z = self.to_mu(x)
        s = self.strecth(z)
        return s


    def latent(self, x):
        z = self.encode(x)
        return z

    def reparameterize(self, z):
        diff = torch.abs(z - z.unsqueeze(axis = 1))
        none_zeros = torch.where(diff == 0., torch.tensor([100.]).to(z.device), diff)    
        z_scores,_ = torch.min(none_zeros, axis = 1)
        std =  torch.normal(mean = 0., std = 1.*z_scores).to(z.device)
        s = z + std
        c = torch.cat((torch.cos(2*np.pi*s), torch.sin(2*np.pi*s)), 0)
        c = c.T.reshape(self.latent_dim*2,-1).T
        return c


    def forward(self, x):
        z = self.encode(x)
        z2 = self.reparameterize(z)
        reconstr = self.decode(z2)
        return [reconstr, z2, z] 


