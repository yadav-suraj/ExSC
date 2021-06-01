import torch
import torchvision
import torchvision.transforms as transforms
import plotly.express as px
import seaborn as sns
import os, sys

from src.metric import *


##############################################################
###################### Data utilities ########################
##############################################################

def get_data(params):
    if params['dset'] == 'mnist':
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.1307, 0.3081)])
        train_dataset = torchvision.datasets.MNIST(root=params['data_dir'], train=True, transform=transform, download = True)
        test_dataset = torchvision.datasets.MNIST(root=params['data_dir'], train=False, transform=transform, download = True)
        
    elif params['dset'] == 'fmnist':
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5, 0.5)])
        train_dataset = torchvision.datasets.FashionMNIST(root=params['data_dir'], train=True, transform=transform, download = True)
        test_dataset = torchvision.datasets.FashionMNIST(root=params['data_dir'], train=False, transform=transform, download = True)
    else:
        raise Exception("Please choose dset from {'mnist', 'fmnist'}")
    
    batch_size = params['batch_size']
    
    if params['use_all_data']:
        train_dataset = ConcatDataset([train_dataset, test_dataset])
    else:
        train_dataset = train_dataset
        
    #validation set construction
    total = len(train_dataset)
    
    valid_num = int(params['val_set_fraction'] * total)
    train_set, valid_set = torch.utils.data.random_split(train_dataset, [total - (valid_num),valid_num])
    
    train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = batch_size, \
                                               shuffle = True, num_workers = params['num_workers'])
    valid_loader = torch.utils.data.DataLoader(dataset = valid_set, batch_size = 3*batch_size, shuffle = True)
    
    
    assert len(train_set)%batch_size == 0 and len(valid_set)%batch_size == 0
    #needed this since passing a batch of size < batch_size towards the end of each epoch will cause an error of size mismatch of last batch's size with A's size
    #all batches from train_loader in a epoch are needed to be of same size

    return train_set, train_loader, valid_set, valid_loader


##############################################################
##################### Model utilities ########################
##############################################################

#utility functions used to create the layer attributes - #number of input units, activation function
def get_k_list(n, k, kf, num_layers):
    k_list = []
    if kf != None:
        while True:
            k1 = int(n * kf)
            if k1 > k:
                k_list.append(k1)
                n = k1
            else:
                k_list.append(k)
                break
        if num_layers != None:
            print("WARNING: Using kf: "+str(kf)+" to construct the layers. Though num_layers it not None. num_layers: "+str(num_layers))
    elif (num_layers != None) and (num_layers > 0) and (num_layers < 3):
        if num_layers == 1:
            k_list.append(int(n/2.0))
        elif num_layers == 2:
            k_list.append(int(n/2.0))
            k_list.append(int(n/3.0))
        else:
            assert False,"Unexpected values for num_layers: "+str(num_layers)+". Expected; 1 or 2"
        k_list.append(k)
    else:
        assert False,"Unexpected values for kf: "+str(kf)+", num_layers: "+str(num_layers)
    return k_list

def get_actf_list(k_list, e_actf):
    actf_list_e = []
    for k in k_list:
        actf_list_e.append(e_actf) 
    actf_list_e.append(e_actf)  #for the extra decoding layer
    return actf_list_e


##############################################################
################### Training utilities #######################
##############################################################

def stopping_criteria(five_loss_val,threshold, first_time):
    if first_time:
        diff = []
        for i in range(1,len(five_loss_val)):
            diff.append(abs(five_loss_val[i]-five_loss_val[i-1]))
        if max(diff) <= threshold:
            return True
    else:
        if abs(five_loss_val[-1] - five_loss_val[-2]) <= threshold:
            return True
    
    return False

##############################################################
################### Inference utilities ######################
##############################################################

def plots(path, enc_C, mu):
    #getting some plots to check orthogonality
    normed = normalize(mu)
    
    if not os.path.exists(path):
        os.makedirs(path)
    fig1 = px.imshow(enc_C.T@enc_C)
    fig1.write_image(path + "encoding.png")
    fig2 = px.imshow(mu.T@mu)
    fig2.write_image(path + "mu.png")
    fig2 = px.imshow(normed.T@normed)
    fig2.write_image(path + "normed_mu.png")


def inferenceCallback(entire_dataset, model, n_clusters, foldername = "sample_output/"):
    
    model = model.to('cpu')
    model.eval()

    NMI_mu = {}
    with torch.no_grad():
        print('When entire training data is taken\n')

        for X_train,Y_train in entire_dataset:

            X_train = X_train.reshape(-1,784)
            print("Number of points:", X_train.shape[0])
            enc_C, dec_C, mu, logvar, logspike = model.inference(X_train)
            
            path = "./images/"+foldername
            plots(path, enc_C, mu)
                
            nmi_pred = get_pred(mu, n_clusters)
            
            try:
                nmi_enc_pred = get_pred(enc_C, n_clusters)
                acc_enc = acc(torch.tensor(nmi_enc_pred), Y_train.cpu())
                NMI_enc = NMI_score(nmi_enc_pred,Y_train.cpu())
                print("NMI_enc over entire training data: {:.4f}; ACC_enc : {:.4f}".format(NMI_enc,acc_enc))
                
            except Exception as e:
                print("enc_C has an error while making predictions; Error: ",e)
            
            print("\nFor mu: ")
            print_accuracy(nmi_pred, Y_train.cpu().numpy())
            acc_mu = acc(torch.tensor(nmi_pred), Y_train.cpu())
            NMI_mu = NMI_score(nmi_pred,Y_train.cpu())
            #print("NMI(mu) for batch {}: {:.5f} and NMI(new_mu): {:.5f}".format(i+1,NMI_mu[i+1],NMI_new[i+1]))

        print("NMI_mu over entire training data: {:.4f}; ACC_mu : {:.4f}".format(NMI_mu,acc_mu))
        print('\n\n')
