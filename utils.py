import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import sys
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans,SpectralClustering
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from sklearn.metrics import pairwise_distances
from sklearn.metrics import confusion_matrix
from munkres import Munkres
from scipy.optimize import linear_sum_assignment as linear_assignment

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
  
def normalize(X):
    #n_cols = X.shape[1]
    #for i in range(n_cols):
    #    factor = torch.sum(torch.pow(X[:,i],2))
    #    factor = torch.sqrt(factor)
    #    X[:,i] = X[:,i]/factor
    factor = torch.sqrt(torch.diag(torch.mm(torch.transpose(X,1,0),X)))
    X_normed = X / factor
    return X_normed,factor

def spec_loss(Y,L):
    #Alternatively:
    #Dy = pairwise_distances(Y)
    #return torch.sum(affinity * Dy) / 2
    return torch.trace(torch.mm(torch.mm(Y.T,L),Y))

def cayley_map(X):
    n = X.size(0)
    Id = torch.eye(n, dtype=X.dtype, device=X.device)
    return torch.solve(Id - X, Id + X)[0]

def get_pred(Y_i,n_clusters):
    
    #normalizing each entry by the root over squared sum of column entries
    Y_i = Y_i.cpu()
    Y_norm,_ = normalize(Y_i)    #refer to algorithm in "Tutorial on Spectral Clustering"
    #print(Y_norm.shape)

    kmeans = KMeans(n_clusters = n_clusters,random_state = 0).fit(Y_norm)
    pred_labels = kmeans.labels_
    #print("\nPredicted labels:\n",pred_labels)
    return pred_labels


def NMI_score(pred_labels,Y_train):
    true_labels = np.array(Y_train.cpu())
    #print("\nTrue labels:\n",true_labels)

    nmi_score = NMI(true_labels,pred_labels)
    
    return nmi_score

def get_y_preds(cluster_assignments, y_true, n_clusters):
    '''
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)
    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset
    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    '''
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred, confusion_matrix

def get_accuracy(cluster_assignments, y_true, n_clusters):
    '''
    Computes the accuracy based on the provided kmeans cluster assignments
    and true labels, using the Munkres algorithm
    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset
    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    '''
    y_pred, confusion_matrix = get_y_preds(cluster_assignments, y_true, n_clusters)
    # calculate the accuracy
    return np.mean(y_pred == y_true), confusion_matrix

def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))

    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:,j]) # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i,j]
            cost_matrix[j,i] = s-t
    return cost_matrix

def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    clusterLabels = np.zeros(n_clusters)
    for i in range(n_clusters):
        clusterLabels[i] = indices[i][1]
    return clusterLabels

def print_accuracy(cluster_assignments, y_true, n_clusters = 10, extra_identifier=''):
    '''
    Convenience function: prints the accuracy
    '''
    # get accuracy
    accuracy, confusion_matrix = get_accuracy(cluster_assignments, y_true, n_clusters)
    # get the confusion matrix
    print('confusion matrix{}: '.format(extra_identifier))
    print(confusion_matrix)
    print('accuracy: '.format(extra_identifier) + str(np.round(accuracy, 3)))

def acc(y_pred, y_target):
    D = max(y_pred.max(), y_target.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size(0)):
        w[y_pred[i], y_target[i]] += 1

    ind = linear_assignment(w.max() - w)
    return sum(w[i, j] for i, j in zip(ind[0],ind[1])) * 1.0 / y_pred.size(0)