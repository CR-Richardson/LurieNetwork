import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize



"""
Implementation of SVD Combo Net from 'RNNs of RNNs: Recursive construction of stable assemblies of recurrent neural 
networks', Kozachkov et. al. (2022).
"""


class Positive(nn.Module):
    """
    For mapping an unconstrained tensor to a tensor with all positive elements.
    """

    def __init__(self):
        super().__init__()
        self.e = 1e-5
        
    def forward(self, Z):
        return torch.abs(Z) + self.e 



class Pos_Diag(nn.Module):
    """
    For mapping an unconstrained (n,n) matrix to a positive diagonal matrix.
    args:
        n: Dimension of state.
    """
    
    def __init__(self, n:int): # n=input m=output
        super().__init__()
        self.e = 1e-5
        self.mask = torch.eye(n)
        
    def forward(self, Z):
        return self.mask * ( torch.abs(Z) + self.e )
       
        
        
class Bounded_PositiveDiag(nn.Module):
    """
    For mapping an unconstrained (n,n) matrix to a positive diagonal matrix with upper bound 1/g.
    args:
        n: Dimension of state.
        g: Reciprocal of the upper bound.
    """
    
    def __init__(self, n:int, g:float):
        super().__init__()
        self.mask = (1/g) * torch.eye(n)
        
    def forward(self, Z):
        return self.mask * torch.exp(-Z**2)



class Skew(nn.Module):
    """
    For mapping an unconstrained square matrix to a skew-symmetric matrix.
    """
    
    def forward(self, Z):
        return Z.triu(1) - Z.triu(1).transpose(-1, -2)



class MatrixExponential(nn.Module):
    """
    For mapping a square matrix through the matrix exponential function.
    If the input matrix is symmetric, output matrix is positive definite. 
    If the input matrix is skew-symmetric, output matrix is orthogonal. 
    """

    def forward(self, Z):
        return torch.matrix_exp(Z)        
        


class SVD_Combo(nn.Module):
    """
    SVD Combo Network.
    args:
            n: Dimension of state.
         tmax: Number of time steps to loop through.
         step: Integration step size.
            g: Slope restriction on nonlinearity.
           bx: Bias term which is learnt if set to None, or can be specified a priori and passed as input.
    returns:
            X: Predictions for a given batch with size (bs, tmax, n).
    """
    
    def __init__(self, n:int, tmax:int, step:float, g:float, bx):
        super().__init__()
        self.n = n
        self.g = g
        self.tmax = tmax
        self.step = step

        self.A_ = nn.Linear(1, 1, bias=False, dtype=torch.double)
        self.Theta =  nn.Linear(n, n, bias=False, dtype=torch.double)
        self.U = nn.Linear(n, n, bias=False, dtype=torch.double)
        self.Sigma = nn.Linear(n, n, bias=False, dtype=torch.double)
        self.V = nn.Linear(n, n, bias=False, dtype=torch.double)

        parametrize.register_parametrization(self.A_, "weight", Positive())
        parametrize.register_parametrization(self.Theta, "weight", Pos_Diag(n))
        parametrize.register_parametrization(self.U, "weight", Skew())
        parametrize.register_parametrization(self.U, "weight", MatrixExponential())
        parametrize.register_parametrization(self.Sigma, "weight", Bounded_PositiveDiag(n,g))
        parametrize.register_parametrization(self.V, "weight", Skew())
        parametrize.register_parametrization(self.V, "weight", MatrixExponential())

        if bx == None:
            self.bx_ind = True
            self.bx = nn.Linear(1,n, bias=False, dtype=torch.double)
            nn.init.zeros_(self.bx.weight)
        else:
            self.bx_ind = False
            self.bx = bx



    def get_AB(self):
        """
        Function for computing SVD Combo Network parameters (A, B) from the variables initialised in init().
        """

        self.A = - self.A_.weight * torch.eye(self.n, dtype=torch.double)
        self.B = torch.matmul( self.V.weight.t(),  self.Theta.weight )
        self.B = torch.matmul( self.Sigma.weight, self.B )
        self.B = torch.matmul( self.U.weight, self.B )
        self.B = torch.matmul( torch.linalg.inv(self.Theta.weight), self.B )



    def get_P(self):
        """
        Function for computing P from Theta.
        """

        self.P = torch.matmul(self.Theta.weight, self.Theta.weight)



    def forward(self, X0):

        bs, n = X0.shape

        self.get_AB()

        if self.bx_ind == True:
            bx_tiled = torch.tile( self.bx.weight, (1,bs) )
        else:
            bx_tiled = torch.tile( self.bx, (1,bs) )
        
        X = []
        Xi = X0.t() # (n, bs)
        X.append(Xi)
    
        for i in range(self.tmax-1):
            Xi = Xi + self.step * ( torch.matmul( self.A, Xi )  + torch.matmul( self.B, torch.relu(Xi)  ) + bx_tiled )
            X.append(Xi)

        X = torch.stack(X, dim=0) # (tmax, n, bs)
        X = torch.transpose(X,0,2) # (bs, n, tmax)
        
        return torch.transpose(X,1,2) # (bs, tmax, n)


