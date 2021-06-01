import torch
import torch.nn as nn
import numpy as np

def cayley_map(A):
    n = A.shape[0]
    Id = torch.eye(n, dtype=A.dtype , device = A.device)
    halfA = 0.5*A
    cay = torch.solve(Id + halfA, Id - halfA).solution
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
    
#Below designed class hasn't been tested properly yet for results; can ignore
class cayleylayer(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        
        self.A = torch.Tensor(size,size)
        self.cayley_init_(self.A)
        self.weights = nn.Parameter(self.cayley_map())
        bias = torch.Tensor(size)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        #nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init
        
    def cayley_map(self):
        n = self.A.shape[0]
        Id = torch.eye(n, dtype=self.A.dtype, device=self.A.device)
        cay = torch.solve(Id - self.A,Id + self.A)[0]
        return cay

    def cayley_init_(self,A):
        '''For Initializing A as skew symmetric matrix'''
        size = A.size(0) // 2
        diag = A.new(size).uniform_(0., np.pi / 2.)
        diag = -torch.sqrt((1. - torch.cos(diag))/(1. + torch.cos(diag)))
        return self.create_diag_(A, diag)
    
    @staticmethod
    def create_diag_(A, diag):
        n = A.size(0)
        diag_z = torch.zeros(n-1)
        diag_z[::2] = diag
        A_init = torch.diag(diag_z, diagonal=1)
        A_init = A_init - A_init.T
        with torch.no_grad():
            A.copy_(A_init)
            return A

    def forward(self, x):
        w_times_x= torch.mm(x, self.weights.t())
        return torch.add(w_times_x, self.bias)  # w times x + b
