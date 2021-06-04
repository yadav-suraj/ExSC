import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import datetime
import collections
import pprint
import sys, os
import argparse

from collections import defaultdict
from sklearn.cluster import KMeans
from torch.optim import lr_scheduler
#from tqdm import tqdm_notebook as tqdm

from src.model import get_model
from src.utils import get_data, inferenceCallback, stopping_criteria
from src.metric import normalize
from src.affinity import laplacian

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, help='gpu number to use', default='')
parser.add_argument('--dset', type=str, help='dataset to use from {mnist,fmnist}', default='mnist')
args = parser.parse_args()

# SELECT GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

params = defaultdict(lambda: None)

# SET GENERAL HYPERPARAMETERS
general_params = {
        'dset': args.dset,                  # dataset: mnist/ fmnist (FashionMNIST)
        'val_set_fraction': 0.1,            # fraction of training set to use as validation
        'max_epochs': 1000,                    # maximum number of training epochs
        'infer_epochs' : 5,                 # Number of epoch before calculating NMI during training (just to keep a check; not required)
        'data_dir': './data',               # directory with all data files
        'num_workers': 32,                  # dataloader parameter
        'prior': 'sns',                     # sns = spike and slab; normal = gaussian
        'imagefoldername': 'foldername/'
        }
params.update(general_params)

if params['prior'] == 'sns':
    prior_params = {
        'alpha' : 0.5,
        'c' : 50,
        'c_delta' : 0.001,
        'beta' : 0.1,
        'beta_delta' : 0,
                    }
    params.update(prior_params)
elif params['prior'] == 'normal':
    params.update({'beta' : 20})
    

# SET DATASET SPECIFIC HYPERPARAMETERS
dset_params = {
    'input_size': 784,                  # number of input features
    'n_clusters': 10,                   # number of clusters in data
    'n_nbr': 75,                        # number of nonzero entries (neighbors) to use for graph Laplacian affinity matrix
    'scale_nbr': 2,                     # neighbor used to determine scale of gaussian graph Laplacian; calculated by
                                        # taking median distance of the (scale_nbr)th neighbor, over a set of size batch_size
                                        # sampled from the datset
    'learning_rate': 1e-4,              # optimizer's initial learning rate
    'weight_decay': 1e-4,               # optimizer's weight decay
    'batch_size': 100,                  # batch size
    'actf': 'tanh',                     # activation function
    'early_stop_threshold': 0.1,
    'use_all_data': False,              # enable to use all data for training (no test set)
                }
params.update(dset_params)

    
def main():
    
    start_time = datetime.datetime.now()
    print(f"Starting time of script is {start_time}\n\n")
    
    print("Parameters:")
    pprint.pprint(dict(params))
    
    input_size = params['input_size']
    batch_size = params['batch_size']
    first_time = True
    
    train_set, train_loader, valid_set, valid_loader = get_data(params)
        
    model = get_model(params).to(device)
    params_list = model.parameters()
    optimizer = torch.optim.Adam(params_list, lr= params['learning_rate'], weight_decay= params['weight_decay'])
    
    spec_loss_list, loss_list, valid_loss_list = {}, {}, {}
    early_stop_threshold = params['early_stop_threshold']
    count = 0 
    
    
    # Training loop
    for epoch in range(params['max_epochs']):

        running_loss, running_valid_loss = 0.0, 0.0
        
        if (epoch+1) > 800:
            count += 1
            if count >= 50:
                early_stop_threshold *= 2
                count = 0

        for i,(X_train,Y_train) in enumerate(train_loader):

            X_train  = X_train.view(-1,input_size).to(device)
            L,affinity = laplacian(X_train, params['n_nbr'], params['scale_nbr'])

            optimizer.zero_grad()

            x_enc, x_dec, mu_new, logvar, logspike = model(X_train)
            normed_mu = normalize(mu_new)

            loss = model.loss_function(normed_mu, L, X_train, x_dec, mu_new, logvar, logspike)
            loss.backward(retain_graph=True)
            
            model.update_()    #part of spike and slab prior
            optimizer.step()

            running_loss += loss.item()
        
        loss_epoch = running_loss/len(train_loader)
        loss_list[epoch+1] = loss_epoch
        
        # Validation loss calculation
        for i,(X_valid,Y_valid) in enumerate(valid_loader):
            
            X_valid  = X_valid.view(-1,input_size).to(device)
            L,_ = laplacian(X_valid, params['n_nbr'], params['scale_nbr'])
            
            x_enc, x_dec, mu, logvar, logspike = model.inference(X_valid)
            normed_mu = normalize(mu)
            
            running_valid_loss = model.loss_function(normed_mu, L, X_valid, x_dec, mu, logvar, logspike)
        
        valid_loss_epoch = running_valid_loss / len(valid_loader)
        valid_loss_list[epoch+1] = valid_loss_epoch
        
        if (epoch+1)%params['infer_epochs'] == 0:
            cpy = mu_new.clone().detach().cpu()
            cpy = normalize(cpy)
            ort = torch.mm(cpy.T,cpy) - torch.eye(cpy.shape[1])
            orth_norm = torch.trace(torch.mm(ort.T,ort))    #just calculating on the last batch (could also be done for each batch)
            
            print("Epoch:{}, Loss: {:.4f}, validation set loss:{:.4f}, orthogonality index: {:.5f}".format(epoch+1, loss_epoch, valid_loss_epoch, orth_norm))
        if epoch>800:
            if stopping_criteria(list(valid_loss_list.values())[-5:], early_stop_threshold, first_time = first_time):
                print("\n Early Stopping with early stopping threshold = {} \n".format(early_stop_threshold))
                break
            first_time = False
    
    # Trained model prediction over entire dataset
    entire_dataset = torch.utils.data.DataLoader(dataset = train_set, batch_size = len(train_set))
    inferenceCallback(entire_dataset, model, params['n_clusters'], params['imagefoldername'])
    
    end_time = datetime.datetime.now()
    print(f"\n\nEnding time of script is {end_time}")
    
if __name__ == "__main__":
    main()
