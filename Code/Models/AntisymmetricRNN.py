import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize



"""
Implementation of Antisymmetric RNN from 'ANTISYMMETRICRNN: A DYNAMICAL SYSTEM VIEW
ON RECURRENT NEURAL NETWORKS', Chang et. al (2019).
"""



class Skew(nn.Module):
    """
    For mapping an unconstrained square matrix to a skew-symmetric matrix.
    """
    
    def forward(self, Z):
        return Z.triu(1) - Z.triu(1).transpose(-1, -2)
        
        
        
class AntisymmetricRNN(nn.Module):
    """
    Antisymmetric RNN.
    args:
            n: Dimension of state.
         tmax: Number of time steps to loop through.
         step: Integration step size.
           by: Bias term which is learnt if set to None, or can be specified a priori and passed as input.
    returns:
            X: Predictions for a given batch with size (bs, tmax, n).
    """
    
    def __init__(self, n:int, tmax:int, step:float, by):
        super().__init__()

        self.n = n
        self.tmax = tmax
        self.step = step
        self.register_buffer( 'e', 1e-3 * torch.eye(n, dtype=torch.double) ) # "Diffusion" parameter of C matrix.

        self.C = nn.Linear(n,n, bias=False, dtype=torch.double)
        parametrize.register_parametrization(self.C, "weight", Skew())

        if by == None:
            self.by_ind = True
            self.by = nn.Linear(1,n, bias=False, dtype=torch.double)
            nn.init.zeros_(self.by.weight)
        else:
            self.by_ind = False
            self.by = by
        
    def forward(self, X0):

        bs, n = X0.shape
        
        if self.by_ind:
            by_tiled = torch.tile( self.by.weight, (1,bs) )
        else:
            by_tiled = torch.tile( self.by, (1,bs) )       
        
        X = []
        Xi = X0.t() # (n, bs)
        X.append(Xi)
    
        for i in range(self.tmax-1):
            Yi = torch.matmul( self.C.weight - self.e, Xi ) + by_tiled
            Xi = Xi + self.step * torch.tanh(Yi)
            X.append(Xi)

        X = torch.stack(X, dim=0) # (tmax, n, bs)
        X = torch.transpose(X,0,2) # (bs, n, tmax)
        
        return torch.transpose(X,1,2) # (bs, tmax, n)


