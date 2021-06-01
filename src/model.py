import torch
import torch.nn as nn
import numpy as np
import collections

from torch.nn import functional as F

from src.utils import get_k_list, get_actf_list
from src.cayley import cayley_map, cayley_init_


class BaseModel(nn.Module):
    
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
    
    def __init__(self,params, k_list, actf_list, is_real, cayley_map, cayley_init_):
        super(BaseModel, self).__init__()
        
        do = 0 #dropout hyperparameter
        self.batch_size = params['batch_size']
        self.input_size = params['input_size']
        self.cayley_map = cayley_map
        
        #encoder construction
        enc_layers_dict = collections.OrderedDict()
        num_enc_layers = len(k_list)
        temp_k_decode = []
        k1 = self.input_size
        for i in np.arange(num_enc_layers):
            k2 = k_list[i]
            temp_k_decode.append((int(k1), int(k2)))
            enc_layers_dict["enc-"+str(i)] = nn.Linear(int(k1), int(k2))
            enc_layers_dict["bat-"+str(i)] = nn.BatchNorm1d(int(k2))
            enc_layers_dict["act-"+str(i)] = self.get_actf(actf_list[i])
            enc_layers_dict["drop-"+str(i)] = nn.Dropout(p=do)
            k1 = k2
        
        #mu, var, spike
        hidden_size = int(k2/2.0)
        self.mu_layer = nn.Linear(int(k2), 10, bias=True) #TODO: decide number of output units for mu, sigma
        self.sigma_layer = nn.Linear(int(k2), 10, bias=True)
        self.fc_logspike = nn.Linear(int(k2), 10, bias=True)
        
        #decoder construction
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
        
        self.encoder = nn.Sequential(enc_layers_dict)
        self.decoder = nn.Sequential(dec_layers_dict)
        
        self.A = torch.empty(self.batch_size,self.batch_size)
        #self.A = torch.zeros(self.batch_size,self.batch_size)       #uncomment this for zero initialization based method
        
        self.cayley_init_ = cayley_init_
        self.reset = torch.zero_
        
        self.A = nn.Parameter(self.A)
        self.reset_parameters()                                      #comment this for zero initialization based method
        
    def print_model(self, is_sns = False):
        if is_sns:
            prior = "spike and slab"
        else:
            prior = "normal"
            
        print('='*100+'\n'+'='*100)
        print(f"beta-VAE architecture with {prior} prior\n")
        print("Encoder ")
        print(self.encoder)
        print("="*50)
        print("mu_layer: ")
        print(self.mu_layer)
        print("="*50)
        print("sigma_layer: ")
        print(self.sigma_layer)
        print("="*50)
        if is_sns:
            print("spike_layer")
            print(self.fc_logspike)
            print("="*50)
        print("Decoder ")
        print(self.decoder)
        print('='*100+'\n'+'='*100)
    
    @staticmethod
    def spec_loss(Y,L):
        return torch.trace(torch.mm(torch.mm(Y.T,L),Y))
    
    def reset_parameters(self):
        self.cayley_init_(self.A)
        #self.reset(self.A)
        
    def update_c(self):
        raise NotImplementedError
    
    def update_beta(self):
        raise NotImplementedError
        
    def reparameterize(self, mu, logvar, logspike = None):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
    
    def inference(self, x, is_decoder=True):
        raise NotImplementedError
        
#VAE module
class vae_sns(BaseModel): #VAE
    def __init__(self, params, k_list, actf_list, is_real, cayley_map, cayley_init_):
        super(vae_sns, self).__init__(params, k_list, actf_list, is_real, cayley_map, cayley_init_)
        
        self.print_model(is_sns = True)
        #spike and slab parameters
        self.alpha = params['alpha']
        self.c = params['c']
        self.c_delta = params['c_delta']
        self.beta = params['beta']
        self.beta_delta = params['beta_delta']
        
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
        selection = torch.sigmoid(self.c*(eta + logspike.exp() - 1))
        return selection.mul(gaussian)

    def forward(self, x):
        x_enc_prev = self.encoder(x)
        mu, logvar = self.mu_layer(x_enc_prev), self.sigma_layer(x_enc_prev)
        logspike = -F.relu(-self.fc_logspike(x_enc_prev))
        
        mu_new = torch.mm(self.cayley_map(self.A),mu)
        
        x_enc = self.reparameterize(mu_new, logvar, logspike)
        x_dec = self.decoder(x_enc)
        
        return x_enc, x_dec, mu_new, logvar, logspike
    
    def inference(self, x, is_decoder=True):
        x_enc_prev = self.encoder(x)
        mu, logvar = self.mu_layer(x_enc_prev), self.sigma_layer(x_enc_prev)
        logspike = -F.relu(-self.fc_logspike(x_enc_prev))

        x_enc = self.reparameterize(mu, logvar, logspike)
        x_dec = self.decoder(x_enc)        
        
        if is_decoder:
            x_dec = self.decoder(x_enc)        
        else:
            x_dec = None

        return x_enc, x_dec, mu, logvar, logspike
    
    # vae loss corresponding to the spike and slab prior
    # Reconstruction + KL divergence losses summed over all elements of batch
    def vae_loss(self, x, recon_x, mu, logvar, logspike):
        dim = 1
        
        # Reconstruction term sum (mean?) per batch
        mse_criterion = torch.nn.MSELoss(reduction="none")
        recons_loss = torch.sum(mse_criterion(recon_x,x),dim=dim)
        
        # see Appendix B from VSC paper / Formula 6
        spike = torch.clamp(logspike.exp(), 1e-6, 1.0 - 1e-6) 

        prior1 = -0.5 * torch.sum(spike.mul(1 + logvar - mu.pow(2) - logvar.exp()), dim = dim)
        prior21 = (1 - spike).mul(torch.log((1 - spike) / (1 - self.alpha)))
        prior22 = spike.mul(torch.log(spike / self.alpha))
        prior2 = torch.sum(prior21 + prior22, dim = dim)
        PRIOR = prior1 + prior2

        LOSS = torch.mean(recons_loss + self.beta * PRIOR)

        return LOSS
    
    def loss_function(self, Y, L, *args):
        spectral_loss = self.spec_loss(Y,L)/ Y.shape[0]
        vae_loss = self.vae_loss(*args)
        
        loss = spectral_loss + vae_loss
        return loss
    
    def update_(self):
        # Update value of c gradually 200 ( 150 / 20K = 0.0075 )
        self.update_c()
        self.update_beta()
        
        
class vae_normal(BaseModel):
    def __init__(self,params, k_list, actf_list, is_real, cayley_map, cayley_init_):
        super(vae_normal, self).__init__(params, k_list, actf_list, is_real, cayley_map, cayley_init_)
        
        self.print_model(is_sns = False)
        self.beta = params['beta']
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        logspike = None
        
        x_enc_prev = self.encoder(x)
        
        mu, logvar = self.mu_layer(x_enc_prev), self.sigma_layer(x_enc_prev)
        mu_new = torch.mm(self.cayley_map(self.A),mu)
        
        x_enc = self.reparameterize(mu_new, logvar)
        x_dec = self.decoder(x_enc)        
        
        return x_enc, x_dec, mu_new, logvar, logspike
    
    def inference(self, x, is_decoder=True):
        logspike = None

        x_enc_prev = self.encoder(x)
        mu, logvar = self.mu_layer(x_enc_prev), self.sigma_layer(x_enc_prev)

        x_enc = self.reparameterize(mu, logvar)
        
        if is_decoder:
            x_dec = self.decoder(x_enc)        
        else:
            x_dec = None

        return x_enc, x_dec, mu, logvar, logspike
    
    # Reconstruction + KL divergence losses summed over all elements of batch
    def vae_loss(self, x, recon_x, mu, logvar, is_real):    #TODO: check what dim should be
        dim = 1
        mse_criterion = torch.nn.MSELoss(reduction="none")
        if is_real:
            recons_loss = torch.sum(mse_criterion(recon_x,x),dim=dim)
        else:
            recons_loss = torch.sum(torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='none'),dim=dim)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=dim)
        return torch.mean(recons_loss + (self.beta * KLD))
    
    def loss_function(self, Y, L, *args):
        spectral_loss = self.spec_loss(Y,L)/Y.shape[0]
        vae_loss = self.vae_loss(*args)
        
        loss = spectral_loss + vae_loss
        return loss
    
    def update_(self):
        pass

def get_model(params):
    
    e_actf = params['actf']
    input_size = params['input_size']
    
    # General architecture hyperparameters
    k = 20
    kf = None #0.0000001
    num_layers = 2
    is_real = True
    
    
    k_list = get_k_list(input_size,k,kf,num_layers)
    actf_list = get_actf_list(k_list,e_actf)
    
    if params['prior'] == 'sns':
        model = vae_sns(params, k_list, actf_list, is_real, cayley_map, cayley_init_)
    elif params['prior'] == 'normal':
        model = vae_normal(params, k_list, actf_list, is_real, cayley_map, cayley_init_)
    else:
        raise Exception("wrong selection of prior; choose from sns(for spike and slab) or normal (for standard normal) ")
        
    return model