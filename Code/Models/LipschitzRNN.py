import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize



"""
Implementation of Lipschitz RNN from 'Lipschitz recurrent neural networks', Erichson et. al (2020).
"""



class Exp_01(nn.Module):
    """
    For mapping an unconstrained tensor to a variables with elements in the range [0.5,1].
    """

    def __init__(self):
        super().__init__()
        
    def forward(self, Z):
        return 0.5 * torch.exp(-Z**2) + 0.5
        
        
        
class LipschitzRNN(nn.Module):
    """
    Lipschitz RNN.
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

        self.MA = nn.Linear(n, n, bias=False, dtype=torch.double)
        self.bA = nn.Linear(1, 1, bias=False, dtype=torch.double)
        nn.init.constant_(self.bA.weight, 0.65) # as in Section 7.1
        self.yA = 1e-3 # as in Section 7.1
        
        self.B = torch.eye(n, dtype=torch.double)
            
        self.MW = nn.Linear(n, n, bias=False, dtype=torch.double)
        self.bW = nn.Linear(1, 1, bias=False, dtype=torch.double)
        nn.init.constant_(self.bW.weight, 0.65) # as in Section 7.1
        self.yW = 1e-3 # as in Section 7.1
        
        parametrize.register_parametrization(self.bA, "weight", Exp_01())
        parametrize.register_parametrization(self.bW, "weight", Exp_01())

        if by == None:
            self.by_ind = True
            self.by_ = nn.Linear(1,n, bias=False, dtype=torch.double)
            nn.init.zeros_(self.by_.weight)
        else:
            self.by_ind = False
            self.by_ = by
        
    def forward(self, X0):

        bs, n = X0.shape
        
        self.A =  ( 1 - self.bA.weight ) * ( self.MA.weight + self.MA.weight.t() ) + self.bA.weight * ( self.MA.weight - self.MA.weight.t() ) - self.yA * torch.eye(self.n)

        self.C = ( 1 - self.bA.weight ) * ( self.MW.weight + self.MW.weight.t() ) + self.bW.weight * ( self.MW.weight - self.MW.weight.t() ) - self.yW * torch.eye(self.n)

        if self.by_ind == True:
            self.by = torch.tile( self.by_.weight, (1,bs) )
        else:
            self.by = torch.tile( self.by_, (1,bs) )       
        
        X = []
        Xi = X0.t() # (n, bs)
        X.append(Xi)
    
        for i in range(self.tmax-1):
            Yi = torch.matmul( self.C, Xi ) + self.by
            Xi = Xi + self.step * ( torch.matmul( self.A, Xi )  + torch.matmul( self.B, torch.tanh(Yi)  ) )
            X.append(Xi)

        X = torch.stack(X, dim=0) # (tmax, n, bs)
        X = torch.transpose(X,0,2) # (bs, n, tmax)
        
        return torch.transpose(X,1,2) # (bs, tmax, n)


