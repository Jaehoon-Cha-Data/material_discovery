#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 16:01:44 2023

@author: jaehoon cha
@email: chajaehoon79@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import h5py

def pearson_correlation_matrix(X, Y):
    """
    Computes the Pearson correlation coefficient matrix between two matrices X and Y.
    
    Parameters:
    X: np.ndarray of shape (N, x)
    Y: np.ndarray of shape (N, y)
    
    Returns:
    correlation_matrix: np.ndarray of shape (x, y) containing the Pearson correlation coefficients
    """
    # Center the data by subtracting the mean
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)
    
    # Compute the covariance matrix
    covariance_matrix = np.dot(X_centered.T, Y_centered) / (X.shape[0] - 1)
    
    # Compute the standard deviations
    X_std = np.std(X, axis=0, ddof=1)
    Y_std = np.std(Y, axis=0, ddof=1)
    
    # Compute the correlation matrix
    correlation_matrix = covariance_matrix / np.outer(X_std, Y_std)
    
    return correlation_matrix


def get_outputs(model, data, test_dataloader, recon_exam, lat_dim, dir_name, epoch, device = 'cpu'):    
    model.eval() 
    recon = model.reconstr(recon_exam.to(device))
    nx = 10
    ny = 2

    fig, axs = plt.subplots(1, 10, figsize = (16, 4))        
    for i in range(nx):
        axs[i].plot(recon_exam[i][0].detach().cpu().numpy())
        axs[i].plot(recon[i][0].squeeze().detach().cpu().numpy())
        axs[i].set_xticks([])
        axs[i].set_ylim(0., 1.)


    plt.tight_layout()
    save_name = os.path.join(dir_name, 'reconstr_{}.png'.format(epoch))
    plt.savefig(save_name)    
    plt.close()



    zs = []
    slme = []
    maxs = []
    positions = []
    n_locals = []
    slopes = []
    for tidx, test_data in enumerate(test_dataloader):        
        z = model.latent(test_data['x1'].to(device))
        z = z.detach().cpu().numpy()
        zs.append(z)
        slme.append(test_data['slme'].detach().cpu().numpy())
        maxs.append(test_data['max'].detach().cpu().numpy())
        positions.append(test_data['position'].detach().cpu().numpy())
        n_locals.append(test_data['n_local'].detach().cpu().numpy())
        slopes.append(test_data['slope'].detach().cpu().numpy())
       

    
    zs = np.vstack(zs)
    slme = np.vstack(slme)
    maxs = np.vstack(maxs)
    positions = np.vstack(positions)
    n_locals = np.vstack(n_locals)
    slopes = np.vstack(slopes)


    
    features = np.concatenate((slme, positions, maxs, slopes, n_locals), axis = 1)
    covariance_matrix = pearson_correlation_matrix(features, zs)
    covariance_matrix_abs = np.abs(covariance_matrix)


    
    plt.figure(figsize=(10, 8))
    plt.imshow(covariance_matrix_abs, cmap='coolwarm', aspect='auto')
    
    # Add color bar
    plt.colorbar(label='Covariance')
    
    # Add values on the heatmap
    for i in range(covariance_matrix.shape[0]):
        for j in range(covariance_matrix.shape[1]):
            plt.text(j, i, f'{covariance_matrix[i, j]:.2f}', ha='center', va='center', color='black')
    
    # Add titles and labels
    plt.title('Covariance Matrix with Values')
    plt.xlabel('X features')
    plt.ylabel('Y features')
    plt.xticks(range(covariance_matrix.shape[1]), [r'$Dim_{{{}}}$'.format(i+1) for i in range(lat_dim)])
    plt.yticks(range(covariance_matrix.shape[0]), ['slme', 'p', 'm', 's', 'n'])
    
    save_name = os.path.join(dir_name, 'cov_mat_{}.png'.format(epoch))
    plt.savefig(save_name)
    plt.close()

    
    plt.figure()
    plt.bar(np.arange(lat_dim), np.std(zs, axis = 0))
    plt.title('hist_{}'.format(epoch))
    save_name = os.path.join(dir_name, 'hist_{}.png'.format(epoch))
    plt.savefig(save_name)
    plt.close()
    

    z_prepre = model.latent(recon_exam[-3:-2].to(device)).detach().cpu().numpy()
    # z_prepre = np.mean(zs, axis = 0, keepdims=True)
    z_min = np.percentile(zs,1, axis = 0)
    z_max = np.percentile(zs,99, axis = 0)  


    nx  = 10 # size
    ny = lat_dim


    fig, axs = plt.subplots(ny, nx, figsize = (16, ny*1))   
    for j in range(ny):
        x_values = np.linspace(z_min[j], z_max[j], nx)
        for i in range(nx):
            z_copy = z_prepre.copy()
            z_copy[0, j] = x_values[i]
            x_mean = model.sample(z=torch.Tensor(z_copy).to(device))
            
            axs[j, i].plot(x_mean[0].detach().cpu().numpy().squeeze())
            axs[j, i].set_xticks([])
            axs[j, i].set_ylim(0., 1.)

    plt.tight_layout()
    save_name = os.path.join(dir_name, 'component_{}.png'.format(epoch))
    plt.savefig(save_name)    
    plt.close()

        
    trj = []
    
    j = np.argmax(covariance_matrix_abs, axis = -1)[0]
    fig, axs = plt.subplots(1, nx, figsize = (32, 4))   
    x_values = np.linspace(z_min[j], z_max[j], nx)
    for i in range(nx):
        z_copy = z_prepre.copy()
        z_copy[0, j] = x_values[i]
        x_mean = model.sample(z=torch.Tensor(z_copy).to(device))
        trj.append(x_mean[0].detach().cpu().numpy())
        
        axs[i].plot(x_mean[0].detach().cpu().numpy().squeeze())
        axs[i].set_xticks([])
        axs[i].set_ylim(0., 0.6)

    plt.tight_layout()
    save_name = os.path.join(dir_name, 'component_{}_{}.png'.format(epoch, j))
    plt.savefig(save_name)    
    plt.close()


    trj = np.vstack(trj)
    df = pd.DataFrame(trj)

    # Save to CSV file
    csv_filename = os.path.join(dir_name, 'trj_{}_{}.csv'.format(epoch, j))
    df.to_csv(csv_filename, index=False)




    xs = []
    zs = []
    slme = []
    maxs = []
    positions = []
    n_locals = []
    slopes = []
    materials = []
    dg = []
    for idx in range(data.__len__()//100+1):
        xs.append(data.Spectra[100*idx:100*(idx+1), :])
        inputs = data.Spectra[100*idx:100*(idx+1), :].astype(np.float32)
        inputs = torch.from_numpy(inputs).unsqueeze(1)
        z = model.latent(inputs.to(device))

        z = z.detach().cpu().numpy()
        zs.append(z)
        materials.append(data.material_names[100*idx:100*(idx+1)])
        dg.append(data.direct_gaps_array[100*idx:100*(idx+1)])
        slme.append(data.slmes_array[100*idx:100*(idx+1)])
        maxs.append(data.maxs_array[100*idx:100*(idx+1)])
        positions.append(data.positions_array[100*idx:100*(idx+1)])
        n_locals.append(data.n_locals_array[100*idx:100*(idx+1)])
        slopes.append(data.slopes_array[100*idx:100*(idx+1)])
        
        



    xs = np.concatenate(xs)
    zs = np.concatenate(zs)
    materials = np.concatenate(materials)
    dg = np.concatenate(dg)
    slme = np.concatenate(slme)
    maxs = np.concatenate(maxs)
    positions = np.concatenate(positions)
    n_locals = np.concatenate(n_locals)
    slopes = np.concatenate(slopes)
    
        
    
    
    # Create the HDF5 file
    h5_file_path = os.path.join(dir_name, 'FT_spectra_gap_SLME_v2_with_latent_dim.h5')
    with h5py.File(h5_file_path, 'w') as h5file:
        # Store the material names as a dataset
        h5file.create_dataset('material_names', data=np.array(materials, dtype = 'S'))
        
        # Store alphas, direct gaps, and SLME as separate datasets
        h5file.create_dataset('alphas/cm^-1', data=xs)
        h5file.create_dataset('direct_gaps', data=dg)
        h5file.create_dataset('SLME', data=slme)
        h5file.create_dataset('maxs', data=maxs)
        h5file.create_dataset('positions', data=positions)
        h5file.create_dataset('n_local_maximums', data=n_locals)
        h5file.create_dataset('slopes', data=slopes)
    
    print(f'HDF5 file "{h5_file_path}" created successfully.')





