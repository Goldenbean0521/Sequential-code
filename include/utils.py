# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 12:33:17 2022

@author: Lenovo
"""

import torch
import numpy as np
import torch.autograd as autograd
import math

dtype = torch.FloatTensor

#%%
def inf_train_iter(data_set, batch_size):
    while True:
        select_id = np.random.choice(np.arange(len(data_set)),batch_size,replace=True)
        yield data_set[select_id,:,:]
        
#%%
def calc_gradient_penalty(netD, real_data, fake_data, lam):
    batch_size = real_data.shape[0]
    epsilon = torch.rand(batch_size, 1).type(dtype)
    
    interpolates = epsilon * real_data + ((1 - epsilon) * fake_data)
    interpolates.requires_grad = True
    
    disc_interpolates = netD(interpolates)
    
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, 
                              grad_outputs=torch.ones(disc_interpolates.size()).type(dtype),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    interpolates.requires_grad = False
    
    #take the average of one batch as the gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lam

    return gradient_penalty 
#%%
def initial_fake(dim_z, real_St, vol):
    [batch_size, n_obs, dim_s] = real_St.shape
    init_val = vol*torch.ones(batch_size, dim_s+dim_z).type(dtype)
    init_val[:,0:dim_s] = real_St[:,0,:]
    return init_val   
  
#%%
def simulate(init_val,mu, sigma, corr, dt, dim_s, 
             n_obs, sub_iters=10):
    
    X1=init_val; X2=init_val
    [batch_size,dim_x] = init_val.shape
    dw=math.sqrt(dt)
    data_gen = np.empty([batch_size,n_obs,dim_s])
    data_gen = torch.Tensor(data_gen).type(dtype)
    data_gen[:,0,:] = init_val[:,0:dim_s]
    for i in range(n_obs-1):
        for j in range(sub_iters):
            X2 = X1
            std_BM_increment = torch.randn(batch_size,1,dim_x).type(dtype)
            BM_increment = std_BM_increment*dw
                
            transformed_BM_increment = torch.matmul(BM_increment,
                    corr().type(dtype)).view(batch_size,1,dim_x)
            X1 = X2 + mu(X2)*dt + torch.matmul(transformed_BM_increment, sigma(X2)).view(batch_size,-1)
        data_gen[:,i+1,:] = X1[:,0:dim_s]
    return data_gen

#%%
def simulate_return(init_val, mu, sigma, corr, dt, dim_s,
                    n_obs, sub_iters=10):
    transform_matrix = corr().type(dtype)

    X1=init_val; X2=init_val
    batch_size = init_val.shape[0]
    dim_x = init_val.shape[1]
    dw=math.sqrt(dt);
    data_gen = np.empty([batch_size,n_obs,dim_s])
    data_gen = torch.Tensor(data_gen).type(dtype)
    data_gen[:,0,:] = init_val[:,0:dim_s]
    for i in range(n_obs-1):
        X3 = X1
        for j in range(sub_iters):
            X2 = X1
            std_BM_increment = torch.randn(batch_size,1,dim_x).type(dtype)
            BM_increment = std_BM_increment*dw
                
            transformed_BM_increment = torch.matmul(BM_increment,
                    transform_matrix).view(batch_size,1,dim_x)
            
            X1 = X2 + mu(X2)*dt + torch.matmul(transformed_BM_increment, sigma(X2)).view(batch_size,-1)
        data_gen[:,i+1,:] = X1[:,0:dim_s]-X3[:,0:dim_s]
    return data_gen
#%%
def simulate_big(init_val,mu, sigma, dt, dim_s, 
             n_obs, sub_iters=10):
    
    X1=init_val; X2=init_val
    [batch_size,dim_x] = init_val.shape
    dw=math.sqrt(dt)
    data_gen = np.empty([batch_size,n_obs,dim_s])
    data_gen = torch.Tensor(data_gen).type(dtype)
    data_gen[:,0,:] = init_val[:,0:dim_s]
    for i in range(n_obs-1):
        for j in range(sub_iters):
            X2 = X1
            std_BM_increment = torch.randn(batch_size,1,dim_x).type(dtype)
            BM_increment = std_BM_increment*dw

            X1 = X2 + mu(X2)*dt + torch.matmul(BM_increment, sigma(X2)).view(batch_size,-1)
        data_gen[:,i+1,:] = X1[:,0:dim_s]
    return data_gen

#%%
def simulate_return_big(init_val, mu, sigma, dt, dim_s,
                    n_obs, sub_iters=10):

    X1=init_val; X2=init_val
    [batch_size,dim_x] = init_val.shape
    dw=math.sqrt(dt);
    data_gen = np.empty([batch_size,n_obs,dim_s])
    data_gen = torch.Tensor(data_gen).type(dtype)
    data_gen[:,0,:] = init_val[:,0:dim_s]
    for i in range(n_obs-1):
        X3 = X1
        for j in range(sub_iters):
            X2 = X1
            std_BM_increment = torch.randn(batch_size,1,dim_x).type(dtype)
            BM_increment = std_BM_increment*dw
                            
            X1 = X2 + mu(X2)*dt + torch.matmul(BM_increment, sigma(X2)).view(batch_size,-1)
        data_gen[:,i+1,:] = X1[:,0:dim_s]-X3[:,0:dim_s]
    return data_gen
   