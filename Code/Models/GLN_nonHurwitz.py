import sys
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

# ### settings for running training loop using a specified Google Drive account
from google.colab import drive
drive.mount('/content/gdrive')
gdrive_path = "/content/gdrive/MyDrive/"

root = gdrive_path + ''
sys.path.append(root + 'Code/Models')
import kLurieNet_nonHurwitz as LN_k



"""
Non-Hurwitz Parametrisation of k-contracting Graph Lurie Network.
"""



class L_mask(nn.Module):
  """
  For masking diagonal blocks of L.
  """

  def __init__(self, q, n):
     super().__init__()
     
     mask_block = torch.ones(n, n)
     self.mask = mask_block
     for i in range(q-1):
      self.mask = torch.block_diag(self.mask, mask_block)

     self.mask = torch.ones(q*n, q*n) - self.mask

  def forward(self, Z):
    return Z * self.mask
  


class GLN(nn.Module):
  """
  non-Hurwitz Parametrisation of k-contracting Graph Lurie Network.
    args:
            q: Number of independent Lurie Networks.
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

  def __init__(self, q:int, n:int, tmax:int, step:float, g:float, k:int, A_type:str, B, C, bx, by):
        super().__init__()
        self.q = q
        self.n = n
        self.tmax = tmax
        self.step = step
        self.g = g
        self.k = k
        self.A_type = A_type

        if bx == None:
            self.bx_ind = True
        else:
            self.bx_ind = False

        if by == None:
            self.by_ind = True
        else:
            self.by_ind = False
  
        # instantiate q k-contracting Lurie Networks.
        self.models = []
        for i in range(q):
          self.models.append( LN_k.LurieNet_k(n, tmax, step, g, k, A_type, B, C, bx, by) )

        self.ModList = nn.ModuleList(self.models)


        # instantiate GL for graph coupling matrix, L.
        if q > 1:
            self.GL = nn.Linear(q*n, q*n, bias=False, dtype=torch.double)
            nn.init.zeros_(self.GL.weight)
            parametrize.register_parametrization(self.GL, "weight", L_mask(q, n))
        


  def get_GLN(self):
    """
    Function for computing terms of Graph Lurie Network.
    """

    # Compute individual Lurie Network paramters.
    for i in range(self.q):
      self.models[i].get_ABC()
      self.models[i].get_P()

    # Collect into a single system of independent Lurie Networks.
    self.A = self.models[0].A
    self.B = self.models[0].B
    self.C = self.models[0].C
    self.P = self.models[0].P
    if self.bx_ind:
      self.bx = torch.diag( torch.flatten( self.models[0].bx.weight ) )
    else:
       self.bx = torch.diag( torch.flatten( self.models[0].bx ) )
    if self.by_ind:
       self.by = torch.diag( torch.flatten( self.models[0].by.weight ) )
    else:
       self.by = torch.diag( torch.flatten( self.models[0].by ) )

    for i in range(1,self.q):
      self.A = torch.block_diag(self.A, self.models[i].A)
      self.B = torch.block_diag(self.B, self.models[i].B)
      self.C = torch.block_diag(self.C, self.models[i].C)
      self.P = torch.block_diag(self.P, self.models[i].P)
      if self.bx_ind:
        self.bx = torch.block_diag( self.bx, torch.diag( torch.flatten( self.models[i].bx.weight ) ) )
      else:
         self.bx = torch.block_diag( self.bx, torch.diag( torch.flatten( self.models[i].bx ) ) )
      if self.by_ind:
        self.by = torch.block_diag( self.by, torch.diag( torch.flatten( self.models[i].by.weight ) ) )
      else:
         self.by = torch.block_diag( self.by, torch.diag( torch.flatten( self.models[i].by ) ) )

    self.bx = torch.reshape( torch.diag(self.bx), (self.q*self.n,1) )
    self.by = torch.reshape( torch.diag(self.by), (self.q*self.n,1) )
    self.P = self.P.double()

    # Construct graph coupling matrix, L.
    if self.q == 1:
       self.L = torch.zeros(self.n,self.n)
    else:
        self.L = torch.matmul( self.GL.weight.t(), self.P )
        self.L = torch.matmul( torch.inverse(self.P), self.L )
        self.L = self.GL.weight - self.L



  def forward(self, X0):
    
    bs = X0.shape[0]

    self.get_GLN()
    
    bx_tiled = torch.tile( self.bx, (1,bs) )
    by_tiled = torch.tile( self.by, (1,bs) )

    # Simulate GLN.
    X = []
    Xi = X0.t() # (n, bs)
    X.append(Xi)

    for i in range(self.tmax-1):
        Yi = torch.matmul( self.C, Xi ) + by_tiled
        Xi = Xi + self.step * ( torch.matmul( self.A + self.L, Xi )  + torch.matmul( self.B, torch.tanh(Yi)  ) + bx_tiled )
        X.append(Xi)

    X = torch.stack(X, dim=0) # (tmax, n, bs)
    X = torch.transpose(X,0,2) # (bs, n, tmax)
        
    return torch.transpose(X,1,2) # (bs, tmax, n)