import torch
import torchvision
import numpy as np
import random
import matplotlib.pyplot as plt
import sys, os
import sklearn

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans,SpectralClustering
#from sklearn.metrics import pairwise_distances

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import warnings
warnings.filterwarnings("ignore")

def get_scale(x, n_nbrs):  #Taken off the SpectralNet source code
    '''
    Calculates the scale* based on the median distance of the kth
    neighbors of each point of x*, a m-sized sample of x, where
    k = n_nbrs and m = batch_size

    x:          data for which to compute scale
    batch_size: m in the aforementioned calculation. it is
                also the batch size of spectral net
    n_nbrs:     k in the aforementeiond calculation.

    returns:    the scale*

    *note:      the scale is the variance term of the gaussian
                affinity matrix used by spectral net
    '''
    n = len(x)
    batch_size = x.shape[0]

    # sample a random batch of size batch_size
    sample = x[np.random.randint(n, size=batch_size), :]
    # flatten it
    sample = sample.reshape((batch_size, np.prod(sample.shape[1:])))

    # compute distances of the nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_nbrs).fit(sample)
    distances, _ = nbrs.kneighbors(sample)

    # return the median distance
    return np.median(distances[:, n_nbrs - 1])

def pairwise_distances(x, y=None):
    # from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return torch.clamp(dist, 0.0, np.inf)

def knn_affinity(X, n_nbrs, scale_nbr=None, scale=None, local_scale=None, verbose=False):
    '''
    Calculates the symmetrized Gaussian affinity matrix with k1 nonzero
    affinities for each point, scaled by
    1) a provided scale,
    2) the median distance of the k2th neighbor of each point in X, or
    3) a covariance matrix S where S_ii is the distance of the k2th
    neighbor of each point i, and S_ij = 0 for all i != j
    Here, k1 = n_nbrs, k2=scale_nbr
    X:              input dataset of size n
    n_nbrs:         k1
    scale:          provided scale
    scale_nbr:      k2, used if scale not provided
    local_scale:    if True, then we use the aforementioned option 3),
                    else we use option 2)
    verbose:        extra printouts
    returns:        n x n affinity matrix
    '''
    if isinstance(n_nbrs, np.float):
        n_nbrs = int(n_nbrs)
        
    if scale_nbr == None:
        scale_nbr = 0
        
    #get squared distance
    #Dx = torch.FloatTensor(pairwise_distances(X)**2)
    Dx = pairwise_distances(X)
    #calculate the top k neighbors of minus the distance (so the k closest neighbors)
    nn = torch.topk(-Dx, n_nbrs, sorted = True)

    vals = nn[0]
    # apply scale
    def get_median(scales, m):
        scales = torch.topk(scales, m)[0]
        scale = scales[m - 1]
        return scale, scales
    scales = -vals[:, scale_nbr - 1]
    const = X.shape[0] // 2
    scale, scales = get_median(scales, const)
    vals = vals / (2 * scale)

    #get the affinity
    affVals = torch.exp(vals)
    #flatten this into a single vector of values to shove in a spare matrix
    affVals = torch.flatten(affVals).float() #torch.reshape(affVals, [-1])
    #get the matrix of indexes corresponding to each rank with 1 in the first column and k in the kth column
    nnInd = nn[1]
    #get the J index for the sparse matrix
    jj = torch.reshape(nnInd, (-1, 1))
    #the i index is just sequential to the j matrix
    ii = torch.arange(nnInd.shape[0])
    ii = torch.reshape(ii, (-1, 1))
    ii = torch.tensor(np.tile(ii.numpy(),nnInd.shape[1]))
    ii = torch.reshape(ii, (-1, 1))
    #concatenate the indices to build the sparse matrix
    ii = torch.flatten(ii)
    jj = torch.flatten(jj)
    indices = torch.LongTensor([ii.tolist(),jj.tolist()]) #torch.cat((ii,jj),0).int()
    #assemble the sparse Weight matrix
    W = torch.sparse_coo_tensor(indices,affVals.cpu(), size = Dx.shape).to_dense()

    #symmetrize
    W = ((W+torch.transpose(W,1,0))/2.0).to(device)

    return W

def __gaussian_kernel(x,sigma,is_threshold = False):
    sigma = sigma**2
    dist = pairwise_distances(x)
    
    if is_threshold:
        temp_zeros = torch.zeros_like(dist)
        mu = torch.mean(dist)
        temp_ret = torch.where(dist <= mu,temp_zeros,dist) 
        temp_ret = torch.exp(-0.5 * (temp_ret/sigma))
        return temp_ret.fill_diagonal_(0)
    
    else:
        dist = torch.exp(-0.5 * (dist/sigma))
        return dist.fill_diagonal_(0)


def laplacian(X_train, n_nbrs = None, scale_nbr = None, kind = "unnormed"):
    
    #print("Sigma:",sigma)
    if n_nbrs is not None:
        affinity = knn_affinity(X_train,n_nbrs,scale_nbr = scale_nbr)
    else:
        assert scale_nbr is not None
        np_sigma = get_scale(X_train.cpu(), scale_nbr)
        sigma = torch.tensor(np_sigma).to(device)
        affinity = __gaussian_kernel(X_train,sigma)
        
    #print('\n Affinity matrix:\n',affinity) 
    
    d = torch.sum(affinity,axis = 1)
    D = torch.diag(d)   
    
    #print('\n D:\n',D,'\n')
    
    if kind == "normed":
        D_inv = torch.sqrt(torch.diag(1/d))
        L = D - affinity
        L = torch.mm(torch.mm(D_inv,L), D_inv)  #normalized laplacian
        print("Normed Laplacian is calculated\n")
    
    else:
        L = D - affinity
    
    return L,affinity