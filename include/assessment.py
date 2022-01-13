# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 09:57:14 2022

@author: Lenovo
"""
#%%
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import seaborn

#%%
def visualize(choice, data):
    [observes,n_obs,dim] = data.shape
    if choice == 'real':
        for i in range(observes):
            plt.plot(range(n_obs),data[i,:,0])
    elif choice == 'real log':
        for i in range(observes):
            plt.plot(range(n_obs-1),data[i,1:,0])
    elif choice == 'Heston' or choice == 'polynomial':
        fig = plt.figure(figsize = (17,4))
        ax1 = fig.add_subplot(131)
        for i in range(observes):
            ax1.plot(range(n_obs), data[i,:,0])
        ax2 = fig.add_subplot(132)
        for i in range(observes):
            ax2.plot(range(n_obs), data[i,:,1])
        ax3 = fig.add_subplot(133)
        for i in range(observes):
            ax3.plot(range(n_obs), data[i,:,2])
    else:
        fig = plt.figure(figsize = (17,4))
        ax1 = fig.add_subplot(131)
        for i in range(observes):
            ax1.plot(range(n_obs-1), data[i,1:,0])
        ax2 = fig.add_subplot(132)
        for i in range(observes):
            ax2.plot(range(n_obs-1), data[i,1:,1])
        ax3 = fig.add_subplot(133)
        for i in range(observes):
            ax3.plot(range(n_obs-1), data[i,1:,2])
    plt.show()
    
#%%
def findMDD(X):
    [batch_size, n_obs] = X.shape
    results = np.zeros(batch_size)
    for i in range(batch_size):
        M = -1000; mdd = 0
        for j in range(n_obs):
            if(X[i,j]>M): M=X[i,j]
            temp = (M-X[i,j])/X[i,j]
            if(temp>mdd): mdd= temp
        results[i] = mdd
    return results
#%%
def visualizeMDD(real, fake):
    fig = plt.figure(figsize = (12,4))
    ax1 = fig.add_subplot(121)
    ax1.hist(real,20, density = True, alpha = 0.5,label = 'real')
    ax1.hist(fake,20, density = True, alpha = 0.5,label = 'fake')
    ax1.legend(loc="upper right",fontsize=13)
    ax1.set_xlabel('mdd value')
    ax1.set_ylabel('Density')
    
    ax2 = fig.add_subplot(122)
    line1=seaborn.kdeplot(real,label='real')
    line2=seaborn.kdeplot(fake,label='fake')
    ax2.legend(loc="upper right",fontsize=13)
    ax2.set_xlabel('mdd value')
    
#%%
def trans(data):
    [batch_size, n_obs, dim_s] = data.shape
    result = np.zeros([batch_size,n_obs,dim_s])
    result[:,0,:] = data[:,0,:]
    for i in range(n_obs-1):
        result[:,i+1,:] = result[:,i,:]+data[:,i+1,:]
    result = np.exp(result)
    return result
