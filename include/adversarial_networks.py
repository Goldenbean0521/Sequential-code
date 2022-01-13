# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 11:14:22 2022

@author: Lenovo
"""

import torch
import torch.nn as nn


#%%
class Generator_mu(nn.Module): 
#The output is a 2-dimensional array: length*dim_total (dim_S+dim_V)
    def __init__(self, num_layer, dim_m, dim_in):
        super(Generator_mu, self).__init__()
        dim_out = dim_in
        
        layers = [nn.Linear(dim_in, dim_m), nn.LeakyReLU(0.1, True)]
        for i in range(num_layer-1):
            layers += [nn.Linear(dim_m, dim_m), nn.LeakyReLU(0.1, True)]
        layers += [nn.Linear(dim_m, dim_out)]
        
        self.main = nn.Sequential(*layers)
    def forward(self, Xt):
        return self.main(Xt)
    
#%%
class Generator_sigma(nn.Module): 

#The output is a 2-dimensional array: length*dim_total (dim_S+dim_V)
    def __init__(self, num_layer, dim_m, dim_in):
        super(Generator_sigma, self).__init__()
        dim_out = dim_in
        
        layers = [nn.Linear(dim_in, dim_m), nn.LeakyReLU(0.1, True)]
        for i in range(num_layer-1):
            layers += [nn.Linear(dim_m,dim_m), nn.LeakyReLU(0.1, True)]
        layers += [nn.Linear(dim_m, dim_out)]
        
        main = nn.Sequential(*layers)
        self.main = main
    def forward(self, Xt):
        #We reshape the output into a batch of diagonal matirces
        output = torch.diag_embed(self.main(Xt))
        return output
#%%
class Generator_sigma_big(nn.Module): 

#The output is a 2-dimensional array: length*dim_total (dim_S+dim_V)
    def __init__(self, num_layer, dim_m, dim_in):
        super(Generator_sigma_big, self).__init__()
        self.dim_in = dim_in
        dim_out = dim_in
        layers = [nn.Linear(dim_in, dim_m), nn.LeakyReLU(0.1, True)]
        for i in range(num_layer-1):
            layers += [nn.Linear(dim_m,dim_m), nn.LeakyReLU(0.1, True)]
        layers += [nn.Linear(dim_m, dim_out*dim_out)]
        
        main = nn.Sequential(*layers)
        self.main = main
    def forward(self, Xt):
        #We reshape the output into a batch of diagonal matirces
        output = self.main(Xt).view(-1,self.dim_in,self.dim_in)
        return output
#%%
class Net_corr(nn.Module): 

# transform standard (high-dimensional) BM into correlated BM (but each component is a standard BM)
    def __init__(self, dim_in):
        super(Net_corr,self).__init__()
        corr = torch.eye(dim_in) 
        # corr initialized as identity matrix (standard brownian motion)
        self.corr = torch.nn.Parameter(corr)
        self.register_parameter("corr",self.corr)
    def forward(self):
        #normalize (let the matrix be a correlation matrix)
        diag=torch.diag(torch.matmul(self.corr,self.corr.transpose(0,1)))**(-0.5) #normalizer
        output=torch.matmul(torch.diag_embed(diag),self.corr) # diag(BB^T)^(-1/2)*B
        #return result
        return output
    
#%%
class Discriminator(nn.Module):
    def __init__(self, num_layer, dim_m, dim_in):
        super(Discriminator, self).__init__()
        dim_out = 1
        
        layers = [nn.Linear(dim_in, dim_m), nn.LeakyReLU(0.1, True)]
        for i in range(num_layer-1):
            layers += [nn.Linear(dim_m,dim_m), nn.LeakyReLU(0.1, True)]
        layers += [nn.Linear(dim_m, dim_out)]
        
        main = nn.Sequential(*layers)
        self.main = main
    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)
