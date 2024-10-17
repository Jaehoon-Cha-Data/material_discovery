# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 23:24:17 2021

@author: jaehoon cha
@email: chajaehoon79@gmail.com
"""
import numpy as np 
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import argparse
from collections import OrderedDict
import pickle
import os

from models import *
from dataset import *
from infer import *



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type = str, default = 'spd')
    parser.add_argument('--path_dir', type = str, default = os.getcwd())
    parser.add_argument('--dataset', type = str, default = 'spectra') 
    parser.add_argument('--backbone', type = str, default = 'conv')
    parser.add_argument('--epochs', type = int, default = 1000)
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--lat_dim', type = int, default = 10)
    parser.add_argument('--beta', type = float, default = 4.)
    parser.add_argument('--lr', type = float, default = 1e-4)
    parser.add_argument('--lr_decay', type = float, default = 0.95)
    parser.add_argument('--num_workers', type=int, default = 4)
    parser.add_argument('--rnd', type = int, default = 29152)

    args = parser.parse_args()
    
    config = OrderedDict([
            ('model_name', args.model_name),
            ('path_dir', args.path_dir),
            ('dataset', args.dataset),
            ('backbone', args.backbone),
            ('epochs', args.epochs),
            ('batch_size', args.batch_size),
            ('lat_dim', args.lat_dim),
            ('beta', args.beta),
            ('lr', args.lr),
            ('lr_decay', args.lr_decay),
            ('num_workers', args.num_workers),
            ('rnd', args.rnd),
            ])
    
    return config




config = parse_args()

data_path = os.path.join(config['path_dir'], 'datasets')   

### call data ###    
train_x = Spectra(data_path, transform=transform0)

dataset_size = train_x.__len__()
print(dataset_size)
recon_exam = []
for idx in [6123,  6198, 58, 11650,  7375,   995,  3207,  5243, 14394, 627]:
    img = train_x[idx]['x1']
    recon_exam.append(img)
recon_exam = torch.stack(recon_exam)


train_dataloader = DataLoader(train_x, batch_size= config['batch_size'], shuffle=True, 
                              num_workers=config['num_workers'], pin_memory=True)


test_dataloader = DataLoader(train_x, batch_size= config['batch_size'], shuffle=True)




np.random.seed(config['rnd'])
torch.manual_seed(config['rnd'])  

save_folder = os.path.join(config['path_dir'], 'results')  
try:
    os.mkdir(save_folder)
except OSError:
    pass  

save_folder = os.path.join(save_folder, config['dataset'])  
try:
    os.mkdir(save_folder)
except OSError:
    pass  

save_folder= os.path.join(save_folder, config['model_name'] + '_{}_{}_{}'.format(config['lat_dim'], config['beta'], config['rnd']))
try:
    os.mkdir(save_folder)
except OSError:
    pass  

model_path = os.path.join(save_folder,'model.pth')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")       
W = [[1.]*1 + [0.01]*(config['lat_dim']-1)]    
   
model = SPD(config['lat_dim'], torch.Tensor(W).to(device))

if not os.path.exists(model_path): 
    torch.backends.cudnn.benchmark = True
    model.to(device)    

    
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], betas=(0.9, 0.99))
        
    def train(epoch):
        model.train() 
        epoch_loss = 0
        for iteration, batch in enumerate(train_dataloader, 1):
            input, target = batch['x1'].to(device), batch['x1'].to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = torch.nn.L1Loss()(target, output[0])
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(train_dataloader)))

    
    def checkpoint(epoch):
        model_out_path = os.path.join(save_folder, "model_epoch_{}.pth".format(epoch))
        torch.save(model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))
    
    for epoch in range(config['epochs']):
        train(epoch)
        
        if epoch % 100 == 0:
            checkpoint(epoch)
            get_outputs(model, train_x, test_dataloader, recon_exam, config['lat_dim'], save_folder, epoch,  device = device)

    checkpoint(opt.nEpochs)
    get_outputs(model, train_x, test_dataloader, recon_exam, config['lat_dim'], save_folder, config['epochs'],  device = device)
             
    torch.save(model.state_dict(), save_name)
    

else:
    model.load_state_dict(torch.load(model_path))
    model.to(device)    
    get_outputs(model, train_x, test_dataloader, recon_exam, config['lat_dim'], save_folder, 'infer',  device = device)

    
    