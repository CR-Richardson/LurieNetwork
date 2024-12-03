import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize


"""
Non-Hurwitz Parameterisation of k-contracting Lurie Network.
"""



class Positive(nn.Module):
    """
    For mapping an unconstrained tensor to a tensor with positive elements.
    """

    def __init__(self):
        super().__init__()
        self.e = 1e-5
        
    def forward(self, Z):
        return torch.abs(Z) + self.e 



class Diag(nn.Module):
    """
    For mapping an unconstrained (n,n) matrix to a diagonal matrix.
    args:
        n: Dimension of state.
    """
    
    def __init__(self, n:int): # n=input m=output
        super().__init__()
        self.register_buffer('mask', torch.eye(n, dtype=torch.double))
        
    def forward(self, Z):
        return self.mask * Z



class PositiveDiag(nn.Module):
    """
    For mapping an unconstrained (n,n) matrix to a positive diagonal matrix.
    args:
        n: Dimension of state.
    """
    
    def __init__(self, n:int): # n=input m=output
        super().__init__()
        self.register_buffer('mask', torch.eye(n, dtype=torch.double))
        
    def forward(self, Z):
        return self.mask * torch.abs(Z)



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



def SigmaA(alpha_upp, SA1, GA2, GA3, eye_nsk):
    """
    Computes blocks of the Sigma matrix.
    args:
        alpha_upp: sqrt(4kz) (see paper).
              SA1: diagonal tensor (k-1, k-1). 
              GA2: positive scalar.
              GA3: positive n-k vector.
          eye_nsk: (n-k) identity matrix.
    returns:
                 Block diagonal (n, n) SA matrix.
    """
    
    k = SA1.shape[0] + 1
    n = GA3.shape[0] + k

    if k == 1:
        SA2 = - alpha_upp - GA2
        SA = SA2
    elif k > 1:
        SA2 = - ( alpha_upp + torch.sum( torch.diag(SA1) ) ) - GA2
        SA = torch.block_diag( SA1, SA2 )

    if n > k:
        SA3 = torch.min(SA) * eye_nsk - GA3
        SA = torch.block_diag(SA, SA3)

    return SA

    

class LurieNet_k(nn.Module):
    """
    non-Hurwitz Parametrisation of k-contracting Lurie Network.
    args:
            n: Dimension of state.
         tmax: Number of time steps to loop through.
         step: Integration step size.
            g: Upper bound on slope of nonlinearity.
            k: Integer in [1,n].
       A_type: Denotes which type of A matrix to learn: scalar, diagonal ('diag'), symmetric ('sym') or full.
            B: None if B is to be learnt, or a tensor with dtype double if B is known.
            C: None if C is to be learnt, or a tensor with dtype double if C is known.
           bx: None if bx is to be learnt, or a tensor with dtype double if bx is known.
           by: None if by is to be learnt, or a tensor with dtype double if by is known.
    returns:
            X: Predictions for a given batch with size (bs, tmax, n).
    """

    def __init__(self, n:int, tmax:int, step:float, g:float, k:int, A_type:str, B, C, bx, by):
        super().__init__()
        self.n = n
        self.tmax = tmax
        self.step = step
        self.g = g
        self.k = k
        self.A_type = A_type
        self.e = 1e-3

        if A_type == 'full':
            self.GA_ks1 = nn.Linear(k-1, k-1, bias=False, dtype=torch.double)
            self.GA_k = nn.Linear(1, 1, bias=False, dtype=torch.double)
            self.GA_kp1 = nn.Linear(n-k, n-k, bias=False, dtype=torch.double)
            self.YA = nn.Linear(n, n, bias=False, dtype=torch.double)
            self.UA = nn.Linear(n, n, bias=False, dtype=torch.double)
            nn.init.zeros_(self.GA_ks1.weight)
            nn.init.constant_(self.GA_k.weight, 1e-3)
            parametrize.register_parametrization(self.GA_ks1, "weight", Diag(k-1))
            parametrize.register_parametrization(self.GA_k, "weight", Positive())
            parametrize.register_parametrization(self.GA_kp1, "weight", PositiveDiag(n-k))
            parametrize.register_parametrization(self.YA, "weight", Skew())
            parametrize.register_parametrization(self.UA, "weight", Skew())
            parametrize.register_parametrization(self.UA, "weight", MatrixExponential())
            
        if B == None:
            self.B_ind = True
            self.UB = nn.Linear(n, n, bias=False, dtype=torch.double)      
            self.VB = nn.Linear(n, n, bias=False, dtype=torch.double)
            self.SB = nn.Linear(n, n, bias=False, dtype=torch.double)
            nn.init.constant_(self.SB.weight,1e-3)
            parametrize.register_parametrization(self.UB, "weight", Skew())
            parametrize.register_parametrization(self.UB, "weight", MatrixExponential())
            parametrize.register_parametrization(self.VB, "weight", Skew())
            parametrize.register_parametrization(self.VB, "weight", MatrixExponential())
            parametrize.register_parametrization(self.SB, "weight", PositiveDiag(self.n))  
        else:
            self.B_ind = False
            self.B = B
        
        if C == None:
            self.C_ind = True
            self.UC = nn.Linear(n, n, bias=False, dtype=torch.double)      
            self.VC = nn.Linear(n, n, bias=False, dtype=torch.double)
            self.SC = nn.Linear(n, n, bias=False, dtype=torch.double)
            nn.init.constant_(self.SC.weight,1e-3)
            parametrize.register_parametrization(self.UC, "weight", Skew())
            parametrize.register_parametrization(self.UC, "weight", MatrixExponential())
            parametrize.register_parametrization(self.VC, "weight", Skew())
            parametrize.register_parametrization(self.VC, "weight", MatrixExponential())
            parametrize.register_parametrization(self.SC, "weight", PositiveDiag(self.n))  
        else:
            self.C_ind = False
            self.C = C

        if bx == None:
            self.bx_ind = True
            self.bx = nn.Linear(1,n, bias=False, dtype=torch.double)
            nn.init.zeros_(self.bx.weight)
        else:
            self.bx_ind = False
            self.bx = bx
            
        if by == None:
            self.by_ind = True
            self.by = nn.Linear(1,n, bias=False, dtype=torch.double)
            nn.init.zeros_(self.by.weight)
        else:
            self.by_ind = False
            self.by = by

        self.register_buffer( 'eye_n', torch.eye(self.n, dtype=torch.double) )
        self.register_buffer('eye_nsk', torch.eye(n-k, dtype=torch.double) )

    def get_ABC(self):
        """
        Function for computing Lurie Network parameters (A,B,C) from the variables initialised in init().
        """
    
        if self.C_ind == True:
            self.C = torch.matmul( self.SC.weight,  self.VC.weight.t() )
            self.C = torch.matmul( self.UC.weight, self.C )
            self.sing_C = torch.topk( torch.diag(self.SC.weight), self.k ).values # ordered k largest values of SC
        else:
            self.sing_C = torch.topk( torch.linalg.svdvals(self.C), self.k ).values

        if self.B_ind == True:
            self.B = torch.matmul( self.SB.weight,  self.VB.weight.t() )
            self.B = torch.matmul( self.UB.weight, self.B )
            self.sing_B = torch.topk( torch.diag(self.SB.weight), self.k).values # ordered k largest values of SB
        else:
            self.sing_B = torch.topk( torch.linalg.svdvals(self.B), self.k ).values

        self.alpha_upp = torch.sqrt( 4*self.k *(self.g**2) * torch.sum( (self.sing_B**2) * (self.sing_C**2) ) )
        if self.A_type == 'full':
            self.SA = SigmaA(self.alpha_upp, self.GA_ks1.weight, self.GA_k.weight, self.GA_kp1.weight, self.eye_nsk)
            self.A = torch.matmul( self.SA,  self.UA.weight.t() )
            self.A = torch.matmul( self.UA.weight,  self.A )
            self.A = 0.5*self.A + 0.5*self.YA.weight



    def get_P(self):
        """
        Function for computing P from SA.
        """
        alpha_k = ( 1 / (2*self.k) ) * torch.sum( torch.topk( torch.diag(self.SA), self.k).values )
        self.P = -(1/alpha_k) * self.eye_n



    def forward(self, X0):

        bs, n = X0.shape
        
        self.get_ABC()

        if self.by_ind == True:
            by_tiled = torch.tile( self.by.weight, (1,bs) )
        else:
            by_tiled = torch.tile( self.by, (1,bs) )

        if self.bx_ind == True:
            bx_tiled = torch.tile( self.bx.weight, (1,bs) )
        else:
            bx_tiled = torch.tile( self.bx, (1,bs) )    

        
        X = []
        Xi = X0.t() # (n, bs)
        X.append(Xi)

        for i in range(self.tmax-1):
            Yi = torch.matmul( self.C, Xi ) + by_tiled
            Xi = Xi + self.step * ( torch.matmul( self.A, Xi )  + torch.matmul( self.B, torch.tanh(Yi)  ) + bx_tiled )
            X.append(Xi)

        X = torch.stack(X, dim=0) # (tmax, n, bs)
        X = torch.transpose(X,0,2) # (bs, n, tmax)
        
        return torch.transpose(X,1,2) # (bs, tmax, n)



