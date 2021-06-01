import torch
import torchvision
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from sklearn.metrics import confusion_matrix
from munkres import Munkres
from scipy.optimize import linear_sum_assignment as linear_assignment
#from sklearn.metrics import pairwise_distances

import warnings
warnings.filterwarnings("ignore")


def normalize(X):
    factor = torch.sqrt(torch.diag(torch.mm(torch.transpose(X,1,0),X)))
    X_normed = X / factor
    return X_normed

def get_pred(Y_i,n_clusters):
    
    #normalizing each entry by the root over squared sum of column entries
    Y_i = Y_i.cpu()
    Y_norm = normalize(Y_i)    #refer to algorithm in "Tutorial on Spectral Clustering"
    #print(Y_norm.shape)

    kmeans = KMeans(n_clusters = n_clusters).fit(Y_norm)
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

#Another method to get just accuracy
def acc(y_pred, y_target):
    D = max(y_pred.max(), y_target.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size(0)):
        w[y_pred[i], y_target[i]] += 1

    ind = linear_assignment(w.max() - w)
    return sum(w[i, j] for i, j in zip(ind[0],ind[1])) * 1.0 / y_pred.size(0)