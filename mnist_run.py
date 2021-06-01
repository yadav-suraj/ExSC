import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import plotly.express as px
import datetime
import collections
import sys, os

from utils import *
from torch.optim import lr_scheduler
from torch.nn import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#VAE module
class vae(nn.Module):
  
    def get_actf(self,actf_name):
        if actf_name == "relu":
            A = nn.ReLU()
        elif actf_name == "sigma":
            A = nn.Sigmoid()
        elif actf_name == "tanh":
            A = nn.Tanh()
        elif actf_name == "lrelu":
            A = nn.LeakyReLU()
        elif actf_name == "softmax":
            A = nn.Softmax(dim=1)
        else:
            print("Unknown activation function: ",actf_name)
            sys.exit(1)
        return A
    
    def __init__(self,input_dim, k_list, actf_list, is_real, cayley_map, cayley_init_, device, batch_size):
        super(vae, self).__init__()
        
        do = 0 #dropout hyperparameter
        
        self.cayley_map = cayley_map
        self.device = device
        #enc
        enc_layers_dict = collections.OrderedDict()
        num_enc_layers = len(k_list)
        temp_k_decode = []
        k1 = input_dim
        for i in np.arange(num_enc_layers):
            k2 = k_list[i]
            temp_k_decode.append((int(k1), int(k2)))
            enc_layers_dict["enc-"+str(i)] = nn.Linear(int(k1), int(k2))
            enc_layers_dict["bat-"+str(i)] = nn.BatchNorm1d(int(k2))
            enc_layers_dict["act-"+str(i)] = self.get_actf(actf_list[i])
            enc_layers_dict["drop-"+str(i)] = nn.Dropout(p=do)
            k1 = k2
        #mu, var
        hidden_size = int(k2)
        self.mu_layer = nn.Linear(int(k2), 10, bias=True) 
        self.sigma_layer = nn.Linear(int(k2), 10, bias=True)
        self.fc_logspike = nn.Linear(int(k2), 10, bias=True)
        
        #dec
        dec_layers_dict = collections.OrderedDict()
        i = 0
        dec_layers_dict["dec-"+str(i)] = nn.Linear(10, int(k2),bias=True)
        dec_layers_dict["bat-"+str(i)] = nn.BatchNorm1d(int(k2))
        dec_layers_dict["act-"+str(i)] = self.get_actf(actf_list[i])
        dec_layers_dict["drop-"+str(i)] = nn.Dropout(p=do)
        temp_k_decode.reverse()
        i += 1
        for k_tup in temp_k_decode:
            k1 = k_tup[1]
            k2 = k_tup[0]
            # if i == 0:
            #     dec_layers_dict["dec-"+str(i)] = nn.Linear(int(k1/2.0), int(k2),bias=True)
            # else:
            dec_layers_dict["dec-"+str(i)] = nn.Linear(int(k1), int(k2),bias=True)
            dec_layers_dict["bat-"+str(i)] = nn.BatchNorm1d(int(k2))
            if i == len(temp_k_decode)-1:
                if is_real:
                    dec_layers_dict["act-"+str(i)] = self.get_actf(actf_list[i])
                else:
                    dec_layers_dict["act-"+str(i)] = self.get_actf("sigma")
            else:
                dec_layers_dict["act-"+str(i)] = self.get_actf(actf_list[i])
            dec_layers_dict["drop-"+str(i)] = nn.Dropout(p=do)
            i+=1
        #
        self.encoder = nn.Sequential(enc_layers_dict)
        self.decoder = nn.Sequential(dec_layers_dict)
        #
        print("U: encoder ")
        print(self.encoder)
        print("#")
        print("mu_layer: ")
        print(self.mu_layer)
        print("#")
        print("sigma_layer: ")
        print(self.sigma_layer)
        print("#")
        print("U: decoder ")
        print(self.decoder)
        print('\n\n\n')
        
        self.A = torch.empty(batch_size,batch_size)
        #self.A = torch.zeros(batch_size,batch_size)       #uncomment this for zero initialization based method
        
        self.cayley_init_ = cayley_init_
        self.reset = torch.zero_
        
        self.A = nn.Parameter(self.A)
        self.reset_parameters()                            #comment this for zero initialization based method
        
        #spike and slab parameters
        self.alpha = 0.5
        self.c = 50
        self.c_delta = 0.001
        self.beta = 0.1
        self.beta_delta = 0
        
    def reset_parameters(self):
        self.cayley_init_(self.A)
        #self.reset(self.A)
        
    def update_c(self):
        # Gradually increase c
        self.c += self.c_delta  
    
    def update_beta(self):
        # Gradually adjust beta
        self.beta += self.beta_delta
        
    def reparameterize(self, mu, logvar, logspike):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        gaussian = eps.mul(std).add_(mu)
        eta = torch.rand_like(std)
        selection = F.sigmoid(self.c*(eta + logspike.exp() - 1))
        return selection.mul(gaussian)

    def inference(self,x):
        x_enc_prev = self.encoder(x)
        mu, logvar = self.mu_layer(x_enc_prev), self.sigma_layer(x_enc_prev)
        logspike = -F.relu(-self.fc_logspike(x_enc_prev))

        x_enc = self.reparameterize(mu, logvar, logspike)
        x_dec = self.decoder(x_enc)        

        return x_enc, x_dec, mu, logvar, logspike    
    
    def forward(self, x, is_decoder=True):
        x_enc_prev = self.encoder(x)
        mu, logvar = self.mu_layer(x_enc_prev), self.sigma_layer(x_enc_prev)
        logspike = -F.relu(-self.fc_logspike(x_enc_prev))
        
        mu_new = torch.mm(self.cayley_map(self.A),mu)
        
        x_enc = self.reparameterize(mu_new, logvar, logspike)
        
        if is_decoder:
            x_dec = self.decoder(x_enc)        
        else:
            x_dec = None
            
        return x_dec, mu, mu_new, logvar, logspike
    
    # vae loss corresponding to the spike and slab prior
    # Reconstruction + KL divergence losses summed over all elements of batch
    def loss_function(self, x, recon_x, mu, logvar, logspike, train=False):
        dim = 1
        
        # Reconstruction term sum (mean?) per batch
        flat_input_sz = 784
        
        mse_criterion = torch.nn.MSELoss(reduction="none")
        recons_loss = torch.sum(mse_criterion(recon_x,x),dim=dim)
        
        # see Appendix B from VSC paper / Formula 6
        spike = torch.clamp(logspike.exp(), 1e-6, 1.0 - 1e-6) 

        prior1 = -0.5 * torch.sum(spike.mul(1 + logvar - mu.pow(2) - logvar.exp()), dim = dim)
        prior21 = (1 - spike).mul(torch.log((1 - spike) / (1 - self.alpha)))
        prior22 = spike.mul(torch.log(spike / self.alpha))
        prior2 = torch.sum(prior21 + prior22, dim = dim)
        PRIOR = prior1 + prior2

        LOSS = torch.sum(recons_loss + self.beta * PRIOR)

        return LOSS
    
    
    def update_(self):
        # Update value of c gradually 200 ( 150 / 20K = 0.0075 )
        self.update_c()
        self.update_beta()


#utility functions used to create the layer attributes - #number of input units, activation function

def __get_k_list(n, k, kf, num_layers):
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

def __get_actf_list(k_list, e_actf):
    actf_list_e = []
    for k in k_list:
        actf_list_e.append(e_actf) 
    actf_list_e.append(e_actf)  #for the extra decoding layer
    return actf_list_e

#The beta-VAE loss for gaussian prior
def __get_vae_loss(recon_x, x, mu, logvar, dim, is_real):
    beta = 20
    mse_criterion = torch.nn.MSELoss(reduction="none")
    if is_real:
        recons_loss = torch.sum(mse_criterion(recon_x,x),dim=dim)
    else:
        recons_loss = torch.sum(torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='none'),dim=dim)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=dim)
    return torch.sum(recons_loss + (beta * KLD))

#hyperparameters used to construct the VAE

def cayley_map(A):
    n = A.shape[0]
    Id = torch.eye(n, dtype=A.dtype, device=A.device)
    cay = torch.solve(Id + A,Id - A)[0]
    return cay

def cayley_init_(A):
    '''For Initializing A as skew symmetric matrix'''
    size = A.size(0) // 2
    diag = A.new(size).uniform_(0., np.pi / 2.)
    diag = -torch.sqrt((1. - torch.cos(diag))/(1. + torch.cos(diag)))
    return create_diag_(A, diag)

def create_diag_(A, diag):
    n = A.size(0)
    diag_z = torch.zeros(n-1)
    diag_z[::2] = diag
    A_init = torch.diag(diag_z, diagonal=1)
    A_init = A_init - A_init.T
    with torch.no_grad():
        A.copy_(A_init)
        return A

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

def plots(path, enc_C, mu, logspike):
    #getting some plots to check orthogonality
    normed,_ = normalize(mu)
    
    if not os.path.exists(path):
        os.makedirs(path)
    fig1 = px.imshow(enc_C.T@enc_C)
    fig1.write_image(path + "encoding.png")
    fig2 = px.imshow(mu.T@mu)
    fig2.write_image(path + "mu.png")
    fig2 = px.imshow(normed.T@normed)
    fig2.write_image(path + "normed_mu.png")
    fig4 = px.imshow(logspike.T@logspike)
    fig4.write_image(path + "logspike.png")

def main():
    print("spike and slab prior instead of gaussian")
    print("with dropout 0, mu_new in reparam & kld and sum in vae loss")
    print("\n\n")

    
    input_size = 784
    n_clusters = 10
    val_frac = 0.1

    #Hyperparameters 
    batch_size = 100
    n_nbrs = 75
    scale_nbr = 2
    learning_rate = 1e-4
    weight_decay = 1e-4

    #VAE
    max_epochs = 1000
    infer_epochs = 5 #number of epochs in which inference is done during training. prints out loss values and NMIs. 
    k = 20
    kf = None #0.0000001
    num_layers = 2
    e_actf = "tanh"
    is_real = True

    #utils
    first_time = True #for stopping criteria
    
    train_dataset = torchvision.datasets.MNIST(root='../data', 
                                               train=True, 
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(
                                                       (0.1307,), (0.3081,))]),  
                                               download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True,num_workers = 32)

    test_dataset = torchvision.datasets.MNIST(root='../data',
                                               train = False, 
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(
                                                       (0.1307,), (0.3081,))]),  
                                               download=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True)
    
    entire_trainset = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=len(train_dataset))
    
    entire_testset = torch.utils.data.DataLoader(dataset=test_dataset, 
                                           batch_size=len(test_dataset))

    #validation set construction
    total = len(train_dataset)
    valid_num = int(val_frac * total)
    train_set, valid_set = torch.utils.data.random_split(train_dataset, [total - (valid_num),valid_num])
    train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True, num_workers = 32)
    valid_loader = torch.utils.data.DataLoader(dataset = valid_set, batch_size = 3*batch_size, shuffle = True)
    
    assert len(train_set)%batch_size == 0 and len(valid_set)%batch_size == 0
    #needed this since passing a batch of size < batch_size towards the end of each epoch will cause an error of size mismatch of last batch's size with A's size
    #all batches from train_loader in a epoch are needed to be of same size

    cur_k_list = __get_k_list(input_size,k,kf,num_layers)
    cur_actf_list = __get_actf_list(cur_k_list,e_actf)
    vae_e = vae(input_size,cur_k_list,cur_actf_list,is_real,cayley_map,cayley_init_,device,batch_size).to(device)

    #Training the beta VAE

    params_list = vae_e.parameters()
    optimizer = torch.optim.Adam(params_list, lr=learning_rate, weight_decay=weight_decay)
    #optimizer = torch.optim.RMSprop(params_list, lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 10, verbose = True)
    
    spec_loss_list, loss_list, valid_loss_list = {}, {}, {}
    nmi_mu_list, nmi_new_list = {}, {}
    early_stop_threshold = 1
    count = 0 
    
    for epoch in range(max_epochs):

        running_loss,running_spec_loss,running_nmi,running_nmi_new = 0.0, 0.0, 0.0, 0.0
        running_spec_loss_valid, running_valid_loss = 0.0, 0.0
        
        if (epoch+1) > 300:
            count += 1
            if count >= 50:
                early_stop_threshold *= 2
                count = 0

        for i,(X_train,Y_train) in enumerate(train_loader):

            X_train  = X_train.view(-1,input_size).to(device)
            L,affinity = laplacian(X_train, n_nbrs,scale_nbr)

            optimizer.zero_grad()

            dec_C, mu, mu_new, logvar, logspike = vae_e(X_train)
            normed_mu,_ = normalize(mu_new)
            spectral_loss = spec_loss(normed_mu,L)/mu_new.shape[0]

            loss = spectral_loss + vae_e.loss_function(X_train, dec_C, mu_new, logvar, logspike, train = True)
            loss.backward(retain_graph=True)
            
            vae_e.update_()    #part of spike and slab prior
            
            optimizer.step()

            running_loss += loss.item()
            if (epoch+1)%infer_epochs == 0:
                c = mu.clone().detach().cpu()             
                d = mu_new.clone().detach().cpu()
                
                mu_pred = get_pred(c,n_clusters)
                new_mu_pred = get_pred(d,n_clusters)
                
                nmi_mu = NMI_score(mu_pred,Y_train)
                nmi_new_mu = NMI_score(new_mu_pred,Y_train)
                
                running_spec_loss += spectral_loss.item()
                running_nmi += nmi_mu
                running_nmi_new += nmi_new_mu
                
        
        loss_epoch = running_loss/len(train_loader)
        loss_list[epoch+1] = loss_epoch
        break
        
        #Validation loss on 10% of training dataset
        for i,(X_valid,Y_valid) in enumerate(valid_loader):
            
            X_valid  = X_valid.view(-1,input_size).to(device)
            L_valid,_ = laplacian(X_valid, n_nbrs,scale_nbr)
            
            enc_C, dec_C, mu, logvar, logspike = vae_e.inference(X_valid)
            normed_mu,_ = normalize(mu)
            
            spec_loss_valid = spec_loss(normed_mu,L_valid)/mu.shape[0]
            running_spec_loss_valid += spec_loss_valid.item()
            running_valid_loss = spec_loss_valid + vae_e.loss_function(X_valid, dec_C, mu, logvar, logspike)
            break
        
        valid_loss_epoch = running_valid_loss / len(valid_loader)
        valid_loss_list[epoch+1] = valid_loss_epoch
        
        scheduler.step(valid_loss.item())
        
        if (epoch+1)%infer_epochs == 0:
            spec_loss_epoch = running_spec_loss/len(train_loader)
            nmi_epoch = running_nmi/len(train_loader)
            nmi_mu_epoch = running_nmi_new/len(train_loader)

            spec_loss_list[epoch+1] = spec_loss_epoch
            nmi_mu_list[epoch+1] = nmi_epoch
            nmi_new_list[epoch+1] = nmi_mu_epoch

            d,_ = normalize(d)
            ort = torch.mm(d.T,d) - torch.eye(d.shape[1])
            orth_norm = torch.trace(torch.mm(ort.T,ort))    #just calculating on the last batch (could be done for each batch later)
            
            print("Epoch:{}, Loss: {:.4f}, spectral loss:{:.4f}, validation set loss:{:.4f}, validation set spectral loss:{:.4f}, orthogonality index: {:.5f} NMI_mu: {:.4f}, NMI_new_mu: {:.4f}".format(epoch+1, loss_epoch, spec_loss_epoch, valid_loss.item(), spec_loss_valid.item(), orth_norm, nmi_epoch, nmi_mu_epoch))
        if epoch>400:
            if stopping_criteria(list(valid_loss_list.values())[-5:], early_stop_threshold, first_time = first_time):
                print("\n Early Stopping with early stopping threshold = {} \n".format(early_stop_threshold))
                break
            first_time = False


    vae_e = vae_e.to('cpu')
    vae_e.eval()

    NMI_mu = {}
    NMI_new_mu = {}
    with torch.no_grad():
        print('When entire train dataset is taken\n')

        for X_train,Y_train in entire_trainset:

            X_train = X_train.reshape(-1,784)
            print("Number of points:", X_train.shape[0])
            enc_C, dec_C, mu, logvar, logspike = vae_e.inference(X_train)
            
            path = "./images/directory_name/"
            plots(path, enc_C, mu, logspike)
                
            nmi_pred = get_pred(mu,n_clusters)
            
            try:
                nmi_enc_pred = get_pred(enc_C,n_clusters)
                acc_enc = acc(torch.tensor(nmi_enc_pred), Y_train.cpu())
                NMI_enc = NMI_score(nmi_enc_pred,Y_train.cpu())
                print("NMI_enc over entire train set: {:.4f}; ACC_enc : {:.4f}".format(NMI_enc,acc_enc))
                
                nmi_spike_pred = get_pred(logspike,n_clusters)
                acc_spike = acc(torch.tensor(nmi_spike_pred), Y_train.cpu())
                NMI_spike = NMI_score(nmi_spike_pred,Y_train.cpu())
                print("NMI_spike over entire train set: {:.4f}; ACC_spike : {:.4f}".format(NMI_spike,acc_spike))
                
            except Exception as e:
                print("logspike or enc_C has an error while making predictions; Error: ",e)
            
            print("\nFor mu: ")
            print_accuracy(nmi_pred, Y_train.cpu().numpy())
            acc_mu = acc(torch.tensor(nmi_pred), Y_train.cpu())
            NMI_mu = NMI_score(nmi_pred,Y_train.cpu())
            #print("NMI(mu) for batch {}: {:.5f} and NMI(new_mu): {:.5f}".format(i+1,NMI_mu[i+1],NMI_new[i+1]))

        print("NMI_mu over entire train set: {:.4f}; ACC_mu : {:.4f}".format(NMI_mu,acc_mu))
        print('\n\n')

        print('When entire test dataset is taken\n')

        for X_test,Y_test in entire_testset:

            X_test = X_test.reshape(-1,784)
            print("Number of points:", X_test.shape[0])
            enc_C, dec_C, mu, logvar, logspike = vae_e.inference(X_test)
            
            nmi_pred = get_pred(mu,n_clusters)
            
            acc_mu = acc(torch.tensor(nmi_pred), Y_test.cpu())
            NMI_mu = NMI_score(nmi_pred,Y_test.cpu())
            #print("NMI(mu) for batch {}: {:.5f} and NMI(new_mu): {:.5f}".format(i+1,NMI_mu[i+1],NMI_new[i+1]))

        print("NMI_mu over entire test set: {:.4f}; ACC_mu: {:.4f}".format(NMI_mu,acc_mu))
        print('\n\n')


    print("Loss list:{} \n\n\nSpectral Loss list:{} \n\n\nNMI(mu):{} \n\n\nNMI(new_mu):{} ".format(loss_list, spec_loss_list,nmi_mu_list,nmi_new_list))
    
    #torch.save(vae_e.state_dict(),"./2corrected_model_1000epochs_small_lr.pt")
    
if __name__ == "__main__":
    main()
